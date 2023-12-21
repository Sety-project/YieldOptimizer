import html
import shutil
import sys
import json
import os
from datetime import timedelta, date
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import pickle

import yaml

from scrappers.defillama_history.defillama import FilteredDefiLlama
from research.research_engine import build_ResearchEngine, model_analysis
from strategies.vault_backtest import VaultBacktestEngine
from strategies.cta_betsizing import SingleAssetStrategy
from strategies.cta_backtest import BacktestEngine
from utils.api_utils import extract_args_kwargs

import streamlit as st
import plotly.express as px

from scrappers.defillama_history.coingecko import myCoinGeckoAPI
from utils.io_utils import human_format, check_whitelist, extract_from_paths, dict_list_to_combinations

st.title('Yield optimizer backtest \n by sety-project.eth', help='0xFaf2A8b5fa78cA2786cEf5F7e19f6942EC7cB531')

st.session_state.authentification = "unverified"
# st.session_state.user_tg_handle = ''

with st.spinner('fetching meta data'):
    coingecko = myCoinGeckoAPI()
    coingecko.adapt_address_map_to_defillama()


def prettify_metadata(input_df: pd.DataFrame) -> pd.DataFrame:
    meta_df = deepcopy(input_df)
    meta_df['underlyingTokens'] = meta_df.apply(lambda meta: [coingecko.address_to_symbol(address, meta['chain'])
                                                              for address in meta['underlyingTokens']] if meta[
        'underlyingTokens'] else [],
                                                axis=1)
    meta_df['rewardTokens'] = meta_df.apply(
        lambda meta: [coingecko.address_to_symbol(address.lower(), meta['chain'])
                      for address in meta['rewardTokens']] if meta['rewardTokens'] else [],
        axis=1)
    meta_df['selected'] = True
    return meta_df


def display_download_template_button() -> None:
    # button to download grid template
    pd.DataFrame(
        columns=['strategy.initial_wealth',
                 'run_parameters.models.apy.TrivialEwmPredictor.params.cap',
                 'run_parameters.models.apy.TrivialEwmPredictor.params.halflife',
                 'strategy.cost',
                 'strategy.gas',
                 'strategy.base_buffer',
                 "run_parameters.models.apy.TrivialEwmPredictor.params.horizon",
                 "label_map.apy.horizons",
                 "strategy.concentration_limit",
                 "backtest.end_date",
                 "backtest.start_date"],
        index=[0, 1],
        data=[[1e6, 3, '10d', 5e-4, 50, .2, '28d', [28], .8, date.today().isoformat(), (date.today() - timedelta(days=90)).isoformat()],
              [7e6, 3, '10d', 5e-4, 50, .1, '28d', [28], .4, date.today().isoformat(), (date.today() - timedelta(days=90)).isoformat()]]
    ).to_csv('grid_template.csv')
    with open('grid_template.csv', "rb") as download_file:
        st.download_button(
            label="Download backtest grid template",
            data=download_file,
            file_name='grid_template.csv',
            mime='text/csv',
        )


def display_single_backtest(backtest: pd.DataFrame = None) -> None:
    height = 1000
    width = 1500
    # plot_perf(backtest, override_grid['strategy.base_buffer'][0], height=height, width=width)
    apy = (backtest['apy'] * backtest['weights'].divide(backtest['weights'].sum(axis=1), axis=0)).drop(
        columns='total').reset_index().melt(id_vars='index', var_name='pool', value_name='apy').dropna()
    apyReward = (backtest['apyReward'] * backtest['weights'].divide(backtest['weights'].sum(axis=1),
                                                                            axis=0)).drop(
        columns='total').reset_index().melt(id_vars='index', var_name='pool', value_name='apy').dropna()
    apyReward['pool'] = apyReward['pool'].apply(lambda x: f'reward_{x}')
    apy['apy'] = apy['apy'] - apyReward['apy']
    apy = pd.concat([apy, apyReward], axis=0)
    apy['apy'] = apy['apy'] * 100
    avg_apy = apy.groupby('pool').mean()
    truncated_avg_apy = avg_apy[avg_apy['apy'] > 0.1].reset_index()
    # show avg over backtest
    st.write(f'total apy = {avg_apy["apy"].sum()/100:.1%}')
    st.plotly_chart(
        px.pie(truncated_avg_apy, names='pool', values='apy', title='apy * weights (%)', height=height / 2,
               width=width / 2))
    # show all apy drivers over time
    truncated_apy = apy[apy['pool'].isin(truncated_avg_apy['pool'])]
    st.plotly_chart(
        px.bar(truncated_apy, x='index', y='apy', color='pool', title='apy (%)', height=height, width=width))
    # show all weights over time
    weights = backtest['weights'].drop(columns='total').reset_index().melt(id_vars='index', var_name='pool',
                                                                               value_name='weights').dropna()
    truncated_weights = weights[weights['pool'].isin(truncated_avg_apy['pool'])]
    st.plotly_chart(
        px.bar(truncated_weights, x='index', y='weights', color='pool', title='Allocations ($)', height=height,
               width=width))
    # show all dilutors over time
    dilutor = backtest['dilutor'].drop(columns='total').reset_index().melt(id_vars='index', var_name='pool',
                                                                               value_name='dilutor').dropna()
    dilutor['dilutor'] = 100 * (1 - dilutor['dilutor'])
    truncated_dilutor = dilutor[dilutor['pool'].isin(truncated_avg_apy['pool'])]
    st.plotly_chart(
        px.line(truncated_dilutor, x='index', y='dilutor', color='pool', title='alloc / tvl (%)', height=height,
                width=width))


parameter_keys = ['strategy.initial_wealth',
                  'run_parameters.models.apy.TrivialEwmPredictor.params.cap',
                  'run_parameters.models.apy.TrivialEwmPredictor.params.halflife',
                  'strategy.cost',
                  'strategy.gas',
                  'strategy.base_buffer',
                  "run_parameters.models.apy.TrivialEwmPredictor.params.horizon",
                  "label_map.apy.horizons",
                  "strategy.concentration_limit",
                  "backtest.end_date",
                  "backtest.start_date"]
def display_backtest_grid(grid):
    def display_heatmap(metrics, ind, col, filtering):
        fig = plt.figure(figsize=(20, 20))  # width x height

        # filter
        filtered_grid = grid[np.logical_and.reduce([
            grid[filter_c] == filter_v
            for filter_c, filter_v in filtering.items()])]

        # pivot and display
        for i, values in enumerate(metrics):
            for j, column in enumerate(col):
                df = filtered_grid.pivot_table(values=values, index=ind, columns=col) * 100
                ax = fig.add_subplot(len(col), len(metrics), i + j + 1)
                ax.set_title(f'{values} by {column}')
                sns.heatmap(data=df, ax=ax, square=True, cbar_kws={'shrink': .3}, annot=True,
                            annot_kws={'fontsize': 12})
        st.pyplot(fig=fig)

    metrics = st.multiselect(label='Select metrics', options=['perf', 'churn in d', 'entropy'], default=['perf', 'churn in d'])
    ind = st.selectbox(label='x-axis', options=st.session_state.default_parameters.keys(), index=0)
    col = st.selectbox(label='y-axis', options=st.session_state.default_parameters.keys(), index=4)
    filtering = deepcopy(st.session_state.default_parameters)
    filtering.pop(ind)
    filtering.pop(col)
    try:
        display_heatmap(metrics, ind, col, filtering)
    except Exception as e:
        st.write(str(e))

st.session_state.user_tg_handle = st.sidebar.text_input("Enter your tg handle to backtest:")
if st.session_state.authentification != "verified":
    check_whitelist()

initialize_tab, backtest_tab, analytics_tab, grid_tab, execution_tab = st.tabs(
    ["pool selection", "backtest", "backtest analytics", "grid analytics", "execution helper"])

# load whitelisted protocols from yaml
with initialize_tab:
    with st.form("pool_universe"):
        # instantiate defillama API
        use_oracle = st.selectbox("use_oracle", options=[False, True],
                                  help="this is for depeg snipping. Please use none for now",
                                  disabled=True)
        reference_asset = st.selectbox("reference_asset", options=['usd', 'eth', 'btc'],
                                       help="What asset you are investing")

        top_chains = coingecko.address_map.count().sort_values(ascending=False)[2:23].index
        chains = st.multiselect("chains", options=top_chains, default=['Arbitrum', 'Optimism', 'Polygon'],
                                help="select chains to include in universe")

        tvl_threshold = np.power(10, st.slider("protocol tvl threshold (log10)", min_value=0., value=6., max_value=10.,
                                               step=0.25, help="log10 of the minimum tvl to include in universe"))
        # eg merkle has no tvl
        if tvl_threshold < 2:
            tvl_threshold = -1.
        st.write(f'protocol tvl threshold = {human_format(tvl_threshold)}')
        tvlUsd_floor = np.power(10,
                                st.slider("pool tvl threshold (log10)", min_value=0., value=5., max_value=10.,
                                          step=0.25,
                                          help="log10 of minimum tvl to include in universe"))
        if tvlUsd_floor < 2:
            tvlUsd_floor = -1.
        st.write(f'pool tvl threshold = {human_format(tvlUsd_floor)}')
        if st.form_submit_button("Find pools"):
            st.session_state.defillama = FilteredDefiLlama(reference_asset=reference_asset,
                                                           chains=chains,
                                                           oracle=coingecko,
                                                           protocol_tvl_threshold=tvl_threshold,
                                                           pool_tvl_threshold=tvlUsd_floor,
                                                           use_oracle=use_oracle)
            st.write(
                f'found {len(st.session_state.defillama.pools)} pools among {len(st.session_state.defillama.protocols)} protocols')

    if 'defillama' in st.session_state:
        with st.form("pool_selection"):
            meta_df = prettify_metadata(st.session_state.defillama.pools)
            edited_meta = st.data_editor(meta_df[['selected', 'chain', 'project', 'underlyingTokens', 'tvlUsd', 'apy',
                                                  'apyReward', 'rewardTokens', 'predictedClass', 'binnedConfidence']],
                                         use_container_width=True, hide_index=True)

            if st.form_submit_button("Validate pools"):
                st.session_state.defillama.pools = st.session_state.defillama.pools[edited_meta['selected']]

                with st.status('Fetching data...', expanded=True) as status:
                    if os.path.isdir(os.path.join(os.sep, os.getcwd(), 'data', 'latest')):
                        shutil.rmtree(os.path.join(os.sep, os.getcwd(), 'data', 'latest'))
                    fetch_status = {}
                    st.session_state.all_history = st.session_state.defillama.all_apy_history(
                        dirname=os.path.join(os.sep, os.getcwd(), 'data', 'latest'),
                        status=fetch_status)
                    status.update(
                        label=f'Fetched {len([x for x in fetch_status if fetch_status[x] == "success"])} pools and {len([x for x in fetch_status if fetch_status[x] == "error"])} errors',
                        state='complete', expanded=False)

    if 'all_history' in st.session_state:
        pd.concat(st.session_state.all_history, axis=1).to_csv('data/all_history.csv')
        with open('data/all_history.csv', "rb") as file:
            st.download_button(
                label="Download all history",
                data=file,
                file_name='all_history.csv',
                mime='text/csv',
            )

with backtest_tab:
    if 'all_history' in st.session_state:
        display_download_template_button()

        with st.form("backtest_form"):
            # load default parameters from yaml
            with open(os.path.join(os.sep, os.getcwd(), "config", 'params.yaml'), 'r') as fp:
                parameters = yaml.safe_load(fp)
            st.session_state.default_parameters = extract_from_paths(target=parameters, paths=parameter_keys)

            edit, upload_grid = st.columns(2)
            with edit:
                if st.form_submit_button("Edit parameters"):
                    st.session_state.override_grid = [{'strategy.initial_wealth': np.power(10,
                                                                                           st.slider(
                                                                                               'initial wealth log(10)',
                                                                                               value=6.,
                                                                                               min_value=2.,
                                                                                               max_value=8., step=0.1,
                                                                                               help="log(10) of initial wealth")
                                                                                           ),
                                                      'run_parameters.models.apy.TrivialEwmPredictor.params.cap': 3,
                                                      'run_parameters.models.apy.TrivialEwmPredictor.params.halflife':
                                                          '{}d'.format(
                                                              st.slider('predictor halflife', value=10, min_value=1,
                                                                        max_value=365,
                                                                        help="in days; longer means slower to adapt to new data")),
                                                      'strategy.cost':
                                                          st.slider('slippage', value=5, min_value=0, max_value=100,
                                                                     help="in bps") / 10000,
                                                      'strategy.gas':
                                                          st.slider('gas', value=50, min_value=0, max_value=200,
                                                                     help="avg cost of tx in USD"),
                                                      'strategy.base_buffer':
                                                          st.slider('liquidity buffer', value=10, min_value=0,
                                                                     max_value=25,
                                                                     help="idle capital to keep as buffer, in %") / 100,
                                                      "run_parameters.models.apy.TrivialEwmPredictor.params.horizon":
                                                          "99y",
                                                      "label_map.apy.horizons": [
                                                          st.slider('invest horizon', value=30, min_value=1,
                                                                    max_value=90,
                                                                    help="assumed holding period of investment, in days. This is only used to convert fixed costs in apy inside optimizers")
                                                      ],
                                                      "strategy.concentration_limit":
                                                          st.slider('concentration limit', value=40, min_value=10,
                                                                     max_value=100,
                                                                     help="max allocation into a single pool, in %") / 100,
                                                      "backtest.end_date":
                                                          st.date_input("backtest end", value=date.today()).isoformat(),
                                                      "backtest.start_date":
                                                          st.date_input("backtest start", value=date.today() - timedelta(days=90)).isoformat(),
                                                      }]
                else:
                    st.session_state.override_grid = [st.session_state.default_parameters]

            with upload_grid:
                if uploaded_file := st.file_uploader("Choose a file", type=['csv']):
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.override_grid = pd.read_csv(st.session_state.uploaded_file, index_col=0).to_dict(orient='records')
                    # sorry hack...
                    for record in st.session_state.override_grid:
                        record['label_map.apy.horizons'] = eval(record['label_map.apy.horizons'])

            if st.form_submit_button("Run backtest") and ('override_grid' in st.session_state):

                if 'uploaded_file' in st.session_state:
                    for key, value in st.session_state.override_grid[0].items():
                        st.sidebar.write(f'{key} = {value}')

                if st.session_state.authentification == 'verified':
                    progress_bar1 = st.progress(value=0.0, text='Running grid...')
                    progress_bar2 = st.progress(value=0.0, text='Running backtest...')
                    st.session_state.result = VaultBacktestEngine.run_grid(st.session_state.override_grid,
                                                                           parameters,
                                                                           progress_bar1,
                                                                           progress_bar2)
                    progress_bar1.progress(value=1., text=f'Completed grid')

if 'result' in st.session_state:
    with analytics_tab:
        if selected_run := st.selectbox('Select run name', st.session_state.result['runs'].keys()):
            display_single_backtest(st.session_state.result['runs'][selected_run])
            pd.concat(st.session_state.result['runs'], axis=1).to_csv('logs/runs.csv')
            with open('logs/runs.csv', "rb") as file:
                st.download_button(
                    label="Download runs",
                    data=file,
                    file_name='runs.csv',
                    mime='text/csv',
                )
    if 'uploaded_file' in st.session_state:
        with grid_tab:
            st.dataframe(st.session_state.result['grid'])


with execution_tab:
    st.write("coming soon")

if __name__ == "2__main__":
    args, kwargs = extract_args_kwargs(sys.argv)
    if args[0] in ['backtest', 'grid']:
        # load parameters from yaml
        with open(args[1], 'r') as fp:
            parameters = yaml.safe_load(fp)
        vault_name = args[1].split(os.sep)[-1].split('.')[0]
        dirname = os.path.join(os.sep, os.getcwd(), "logs", vault_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if args[0] == 'grid':
            override_grid = {'strategy.initial_wealth': [1e5, 1e6, 2.5e6, 5e6, 1e7],
                             'run_parameters.models.apy.TrivialEwmPredictor.params.cap': [3],
                             'run_parameters.models.apy.TrivialEwmPredictor.params.halflife': ["10d"],
                             'strategy.cost': [0.0005],
                             'strategy.gas': [0, 20, 50],
                             'strategy.base_buffer': [0.1],
                             "run_parameters.models.apy.TrivialEwmPredictor.params.horizon": ["99y"],
                             "label_map.apy.horizons": [[28]],
                             "strategy.concentration_limit": [0.4]}
            override_grid = dict_list_to_combinations(override_grid)
        else:
            override_grid = dict()
        result = VaultBacktestEngine.run_grid(override_grid, parameters)['grid']

        if args[0] == 'grid':
            result.to_csv(
                os.path.join(os.sep, dirname, 'grid.csv'))

    elif args[0] == 'cta':
        # load parameters
        with open(args[0], 'r') as fp:
            parameters = json.load(fp)

        '''
        model caching for big ML:
            if pickle_override is set, just load pickle and skip
            else run research and print pickle_override if specified
        '''
        outputdir = os.path.join(os.sep, Path.home(), "mktdata", "binance", "results")
        skip_run_research = False
        if "pickle_override" in parameters['input_data']:
            outputfile = os.path.join(os.sep, outputdir, parameters['input_data']["pickle_override"])

            try:
                with open(outputfile, 'rb') as fp:
                    engine = pickle.load(fp)
                    print(f'read {outputfile}')
                    # then fit won't pickle
                    parameters['run_parameters']['pickle_output'] = None
                    skip_run_research = True
            except FileNotFoundError:
                parameters['run_parameters']['pickle_output'] = parameters['input_data']["pickle_override"]

        if not skip_run_research:
            engine = build_ResearchEngine(parameters)

        # backtest
        delta_strategy = SingleAssetStrategy(parameters['strategy'])
        backtest_tab = BacktestEngine(parameters['backtest'], delta_strategy)
        result = backtest_tab.backtest_all(engine)
        result.to_csv(os.path.join(os.sep, outputdir, 'backtest_trajectories.csv'))
        print(f'backtest')

        # analyse perf
        analysis = VaultBacktestEngine.perf_analysis(os.path.join(os.sep, outputdir, 'backtest_trajectories.csv'))
        analysis.to_csv(os.path.join(os.sep, outputdir, 'perf_analysis.csv'))
        print(f'analyse perf')

        # inspect model
        model = model_analysis(engine)
        model.to_csv(os.path.join(os.sep, outputdir, 'model_inspection.csv'))
        print(f'inspect model')
