import sys
import json
import os
from datetime import timedelta, datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import yaml
from plotly.subplots import make_subplots

from scrappers.defillama_history.defillama import FilteredDefiLlama, DiscoveryDefiLlama
from research.research_engine import build_ResearchEngine, model_analysis
from strategies.vault_backtest import VaultBacktestEngine
from strategies.cta_betsizing import SingleAssetStrategy
from strategies.cta_backtest import BacktestEngine
from utils.api_utils import extract_args_kwargs

import streamlit as st
import plotly.express as px

from scrappers.defillama_history.coingecko import myCoinGeckoAPI

def plot_perf(backtest: pd.DataFrame, base_buffer: float) -> None:
    # plot results
    backtest['pnl']["tx_cost"].iloc[0] = 0.001
    cum_tx_cost = backtest['pnl']["tx_cost"].cumsum()
    apy = (backtest['pnl']['wealth'] / backtest['pnl']['wealth'].shift(1) - 1) * 365 / 1 * 100
    max_apy = backtest['full_apy'].max(axis=1) * 100 * (1 - base_buffer)
    max_apy.name = 'max_apy'

    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    # create two independent figures with px.line each containing data from multiple columns
    fig = px.line(apy, render_mode="webgl", )
    fig3 = px.line(max_apy, render_mode="webgl")
    fig3.update_traces(line={"dash": 'dot'})
    fig2 = px.line(cum_tx_cost, render_mode="webgl", )
    fig2.update_traces(yaxis="y2")
    subfig.add_traces(fig.data + fig2.data + fig3.data)
    subfig.layout.xaxis.title = "time"
    subfig.layout.yaxis.title = "apy"
    subfig.layout.yaxis.range = [0, 30]
    subfig.layout.yaxis2.title = "tx_cost"
    subfig.layout.height = 1000
    subfig.layout.width = 2000

    # recoloring is necessary otherwise lines from fig und fig2 would share each color
    # e.g. Linear-, Log- = blue; Linear+, Log+ = red... we don't want this
    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    st.plotly_chart(subfig, use_container_width=True)
    st.write(f'apy mean = {apy.mean()}')
    st.write(f'max apy mean = {max_apy.mean()}')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


st.title('Yield optimizer backtest \n by sety-project.eth', help='0xFaf2A8b5fa78cA2786cEf5F7e19f6942EC7cB531')
coingecko = myCoinGeckoAPI()
coingecko.adapt_address_map_to_defillama()

# load whitelisted protocols from yaml
with st.form("initialize"):
    # instantiate defillama API
    use_oracle = st.selectbox("use_oracle", options=[False, True], help="this is for depeg snipping. Please use none for now",
                 disabled=True)
    reference_asset = st.selectbox("reference_asset", options=['usd', 'eth', 'btc'], help="What asset you are investing")

    top_chains = coingecko.address_map.count().sort_values(ascending=False)[2:23].index
    chains = st.multiselect("chains", options=top_chains, default=['Ethereum'], help="select chains to include in universe")

    tvl_threshold = np.power(10, st.slider("protocol tvl threshold (log10)", min_value=0.5, value=7., max_value=10.,
                                           step=0.1, help="log10 of the minimum tvl to include in universe"))
    st.write(f'protocol tvl threshold = {human_format(tvl_threshold)}')
    tvlUsd_floor = np.power(10,
                            st.slider("pool tvl threshold (log10)", min_value=0.5, value=6., max_value=10., step=0.1,
                                      help="log10 of minimum tvl to include in universe"))
    st.write(f'pool tvl threshold = {human_format(tvlUsd_floor)}')
    if submitted := st.form_submit_button("Find pools"):
        st.session_state.defillama = FilteredDefiLlama(reference_asset=reference_asset,
                                                       chains=chains,
                                                       oracle=coingecko,
                                                       protocol_tvl_threshold=tvl_threshold,
                                                       pool_tvl_threshold=tvlUsd_floor,
                                                       use_oracle=use_oracle)
        st.write(f'found {len(st.session_state.defillama.pools)} pools on {len(st.session_state.defillama.protocols)} protocols')
        st.session_state.initialized = True

if 'initialized' in st.session_state:
    with st.form("data"):
        if submitted := st.form_submit_button("Fetch data"):
            with st.spinner('Fetching data...'):
                all_history = st.session_state.defillama.all_apy_history(dirname=os.path.join(os.sep, os.getcwd(), 'data', 'latest'))
                st.session_state.data_done = True

if 'data_done' in st.session_state:
    with st.form("Run backtest"):
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
        start_date = st.date_input("backtest start", value=end_date - timedelta(days=365))

        if submitted := st.form_submit_button("Run backtest"):
            parameters['backtest']['end_date'] = end_date.isoformat()
            parameters['backtest']['start_date'] = start_date.isoformat()
            with st.spinner('Running backtest...'):
                result = VaultBacktestEngine.run_grid(override_grid, parameters)
                st.session_state.backtest_done = True

if 'backtest_done' in st.session_state:
    filename = "{}/logs/{}/{}__backtest.csv".format(os.getcwd(), 'latest',
                                                                            '_'.join([str(x[0]) for x in override_grid.values()]))
    backtest = pd.read_csv(filename, header=[0, 1], index_col=0)

    plot_perf(backtest, override_grid['strategy.base_buffer'][0])
    st.plotly_chart(px.line(backtest['weights'], title='Weights', height=1000, width=2000))
    st.plotly_chart(px.line(backtest['apy'], title='apy', height=1000, width=2000))



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