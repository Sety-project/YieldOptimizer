import html
import shutil
import sys
import json
import os
from datetime import timedelta, date
from pathlib import Path

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
from utils.io_utils import plot_perf, human_format, check_whitelist

st.title('Yield optimizer backtest \n by sety-project.eth', help='0xFaf2A8b5fa78cA2786cEf5F7e19f6942EC7cB531')

st.session_state.authentification = "unverified"
#st.session_state.user_tg_handle = ''

with st.spinner('fetching meta data'):
    coingecko = myCoinGeckoAPI()
    coingecko.adapt_address_map_to_defillama()

initialize, backtest, analytics = st.tabs(["pool selection", "backtest", "analytics"])
# load whitelisted protocols from yaml
with initialize:
    with st.form("pool_universe"):
        # instantiate defillama API
        use_oracle = st.selectbox("use_oracle", options=[False, True], help="this is for depeg snipping. Please use none for now",
                     disabled=True)
        reference_asset = st.selectbox("reference_asset", options=['usd', 'eth', 'btc'], help="What asset you are investing")

        top_chains = coingecko.address_map.count().sort_values(ascending=False)[2:23].index
        chains = st.multiselect("chains", options=top_chains, default=['Arbitrum', 'Optimism', 'Polygon', 'Avalanche', 'Base'], help="select chains to include in universe")

        tvl_threshold = np.power(10, st.slider("protocol tvl threshold (log10)", min_value=0.5, value=6., max_value=10.,
                                               step=0.1, help="log10 of the minimum tvl to include in universe"))
        st.write(f'protocol tvl threshold = {human_format(tvl_threshold)}')
        tvlUsd_floor = np.power(10,
                                st.slider("pool tvl threshold (log10)", min_value=0.5, value=5., max_value=10., step=0.1,
                                          help="log10 of minimum tvl to include in universe"))
        st.write(f'pool tvl threshold = {human_format(tvlUsd_floor)}')
        if submitted := st.form_submit_button("Find pools"):
            st.session_state.defillama = FilteredDefiLlama(reference_asset=reference_asset,
                                                           chains=chains,
                                                           oracle=coingecko,
                                                           protocol_tvl_threshold=tvl_threshold,
                                                           pool_tvl_threshold=tvlUsd_floor,
                                                           use_oracle=use_oracle)
            st.write(f'found {len(st.session_state.defillama.pools)} pools among {len(st.session_state.defillama.protocols)} protocols')
    if 'defillama' in st.session_state:
        with st.form("pool_selection"):
            meta_df = st.session_state.defillama.pools.copy()
            meta_df['underlyingTokens'] = meta_df.apply(lambda meta: [coingecko.address_to_symbol(address, meta['chain'])
                                                                      for address in meta['underlyingTokens']] if meta[
                'underlyingTokens'] else [],
                                                        axis=1)
            meta_df['rewardTokens'] = meta_df.apply(
                lambda meta: [coingecko.address_to_symbol(address.lower(), meta['chain'])
                              for address in meta['rewardTokens']] if meta['rewardTokens'] else [],
                axis=1)
            meta_df['selected'] = True
            edited_meta = st.data_editor(meta_df[['selected', 'chain', 'project', 'underlyingTokens', 'tvlUsd', 'apy', 'apyReward', 'rewardTokens', 'predictedClass', 'binnedConfidence']],
                         use_container_width=True, hide_index=True)
            if submitted := st.form_submit_button("Validate pools"):
                st.session_state.defillama.pools = st.session_state.defillama.pools[edited_meta['selected']]
                st.session_state.initialized = True

    if 'initialized' in st.session_state:
        with st.status('Fetching data...', expanded=True) as status:
            if os.path.isdir(os.path.join(os.sep, os.getcwd(), 'data', 'latest')):
                shutil.rmtree(os.path.join(os.sep, os.getcwd(), 'data', 'latest'))
            try:
                all_history = st.session_state.defillama.all_apy_history(dirname=os.path.join(os.sep, os.getcwd(), 'data', 'latest'),
                                                                         status=status)
                status.update(label=f'Fetched {len(all_history)} pools', state='complete', expanded=True)
            except Exception as e:
                status.update(label=f'Failed to fetch data: {e}', state='error', expanded=True)
                raise e
            st.session_state.data_done = True

with backtest:
    if 'data_done' in st.session_state:
        with st.form("backtest"):
            # load default parameters from yaml
            with open(os.path.join(os.sep, os.getcwd(), "config", 'params.yaml'), 'r') as fp:
                parameters = yaml.safe_load(fp)

            override_grid = {'strategy.initial_wealth': [np.power(10,
                                                                  st.slider('initial wealth log(10)', value=6., min_value=2., max_value=8., step=0.1, help="log(10) of initial wealth")
                                                                  )],
                             'run_parameters.models.apy.TrivialEwmPredictor.params.cap': [3],
                             'run_parameters.models.apy.TrivialEwmPredictor.params.halflife': ['{}d'.format(
                                 st.slider('predictor halflife', value=10, min_value=1, max_value=365,
                                           help="in days; longer means slower to adapt to new data"))],
                             'strategy.cost':
                                 [st.slider('slippage', value=5, min_value=0, max_value=100, help="in bps") / 10000],
                             'strategy.gas':
                                 [st.slider('gas', value=50, min_value=0, max_value=200, help="avg cost of tx in USD")],
                             'strategy.base_buffer':
                                 [st.slider('liquidity buffer', value=10, min_value=0, max_value=25, help="idle capital to keep as buffer, in %") / 100],
                             "run_parameters.models.apy.TrivialEwmPredictor.params.horizon": ["99y"],
                             "label_map.apy.horizons": [[
                                 st.slider('invest horizon', value=30, min_value=1, max_value=90,
                                           help="assumed holding period of investment, in days. This is only used to convert fixed costs in apy inside optimizers")
                             ]],
                             "strategy.concentration_limit":
                                 [st.slider('concentration limit', value=40, min_value=10, max_value=100,
                                            help="max allocation into a single pool, in %") / 100]
                             }
            st.write('initial_wealth = {}'.format(human_format(override_grid['strategy.initial_wealth'][0])))
            end_date = st.date_input("backtest end", value=date.today())
            start_date = st.date_input("backtest start", value=end_date - timedelta(days=90))
            st.session_state.user_tg_handle = st.text_input("Enter tg handle to backtest:")

            if submitted := st.form_submit_button("Run backtest"):
                parameters['backtest']['end_date'] = end_date.isoformat()
                parameters['backtest']['start_date'] = start_date.isoformat()

                if st.session_state.authentification != "verified":
                    check_whitelist()
                if st.session_state.authentification =='verified':
                    progress_bar = st.progress(value=0.0, text='Running backtest...')
                    result = VaultBacktestEngine.run_grid(override_grid, parameters, progress_bar)
                    progress_bar.progress(value=1., text=f'Completed backtest')
                    st.session_state.backtest_done = True

with analytics:
    if 'backtest_done' in st.session_state:
        filename = "{}/logs/{}/{}__backtest.csv".format(os.getcwd(), 'latest',
                                                                                '_'.join([str(x[0]) for x in override_grid.values()]))
        backtest = pd.read_csv(filename, header=[0, 1], index_col=0)

        height = 1000
        width = 1500

        #plot_perf(backtest, override_grid['strategy.base_buffer'][0], height=height, width=width)
        apy = (backtest['apy'] * backtest['weights'].divide(backtest['weights']['total'], axis=0)).drop(
            columns='total').reset_index().melt(id_vars='index', var_name='pool', value_name='apy').dropna()
        apyReward = (backtest['apyReward'] * backtest['weights'].divide(backtest['weights']['total'], axis=0)).drop(
            columns='total').reset_index().melt(id_vars='index', var_name='pool', value_name='apy').dropna()
        apyReward['pool'] = apyReward['pool'].apply(lambda x: f'reward_{x}')
        apy['apy'] = apy['apy'] - apyReward['apy']
        apy = pd.concat([apy, apyReward], axis=0)
        apy['apy'] = apy['apy']*100

        avg_apy = apy.groupby('pool').mean()
        truncated_avg_apy = avg_apy[avg_apy['apy'] > 0.1].reset_index()

        # show avg over backtest
        st.write('total apy = {}%'.format(int(avg_apy['apy'].sum())))
        st.plotly_chart(px.pie(truncated_avg_apy, names='pool', values='apy', title='apy * weights (%)', height=height/2, width=width/2))

        # show all apy drivers over time
        truncated_apy = apy[apy['pool'].isin(truncated_avg_apy['pool'])]
        st.plotly_chart(px.bar(truncated_apy, x='index', y='apy', color='pool', title='apy (%)', height=height, width=width))

        # show all weights over time
        weights = backtest['weights'].drop(columns='total').reset_index().melt(id_vars='index', var_name='pool',
                                                                                    value_name='weights').dropna()
        truncated_weights = weights[weights['pool'].isin(truncated_avg_apy['pool'])]
        st.plotly_chart(px.bar(truncated_weights, x='index', y='weights', color='pool', title='Allocations ($)', height=height, width=width))

        # show all dilutors over time
        dilutor = backtest['dilutor'].drop(columns='total').reset_index().melt(id_vars='index', var_name='pool',
                                                                     value_name='dilutor').dropna()
        dilutor['dilutor'] = 100 * (1 - dilutor['dilutor'])
        truncated_dilutor = dilutor[dilutor['pool'].isin(truncated_avg_apy['pool'])]
        st.plotly_chart(px.line(truncated_dilutor, x='index', y='dilutor', color='pool', title='alloc / tvl (%)', height=height,width=width))

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

        override_grid = {'strategy.initial_wealth': [1e5, 1e6, 2.5e6, 5e6, 1e7],
                            'run_parameters.models.apy.TrivialEwmPredictor.params.cap': [3],
                          'run_parameters.models.apy.TrivialEwmPredictor.params.halflife': ["10d"],
                          'strategy.cost': [0.0005],
                          'strategy.gas': [0, 20, 50],
                         'strategy.base_buffer': [0.1],
                         "run_parameters.models.apy.TrivialEwmPredictor.params.horizon": ["99y"],
                          "label_map.apy.horizons": [[28]],
                          "strategy.concentration_limit": [0.4]} if args[0] == 'grid' else dict()
        result = VaultBacktestEngine.run_grid(override_grid, parameters)

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
        backtest = BacktestEngine(parameters['backtest'], delta_strategy)
        result = backtest.backtest_all(engine)
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