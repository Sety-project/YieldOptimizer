import os
from copy import deepcopy
from datetime import timedelta, date

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from research.research_engine import FileData
from scrappers.defillama_history.coingecko import myCoinGeckoAPI
from scrappers.defillama_history.defillama import FilteredDefiLlama
from strategies.vault_backtest import VaultBacktestEngine
from utils.io_utils import extract_from_paths, get_date_or_timedelta
from utils.postgres import SqlApi
from utils.streamlit_utils import check_whitelist, human_format, download_grid_template_button, \
    display_single_backtest, download_whitelist_template_button


def prettify_metadata(input_df: pd.DataFrame) -> pd.DataFrame:
    meta_df = deepcopy(input_df)
    try:
        meta_df['underlyingTokens'] = input_df.apply(lambda meta: [coingecko.address_to_symbol(address, meta['chain'])
                                                                  for address in meta['underlyingTokens']] if meta[
            'underlyingTokens'] else [],
                                                    axis=1)
        meta_df['rewardTokens'] = input_df.apply(
            lambda meta: [coingecko.address_to_symbol(address.lower(), meta['chain'])
                          for address in meta['rewardTokens']] if meta['rewardTokens'] else [],
            axis=1)
    except Exception as e:
        print(e)
    meta_df['selected'] = True
    return meta_df

pd.options.mode.chained_assignment = None

st.title('Yield optimizer backtest \n by sety-project.eth', help='0xFaf2A8b5fa78cA2786cEf5F7e19f6942EC7cB531')

st.session_state.authentification = "unverified"
st.session_state.user_tg_handle = st.sidebar.text_input("Enter your tg handle to backtest:")
if st.session_state.authentification != "verified":
    check_whitelist()

if st.session_state.user_tg_handle in st.secrets.admins:
    if st.sidebar.text_input(f"DB to reset (don't !)", key='db_delete'):
        db = SqlApi(st.secrets[st.session_state.db_delete])
        db.delete()
        st.caption(f"deleted {len(db.list_tables())} from {st.session_state.db_delete}")

# load default parameters from yaml
with open(os.path.join(os.sep, os.getcwd(), "config", 'params.yaml'), 'r') as fp:
    parameters = yaml.safe_load(fp)

with st.spinner('fetching meta data'):
    coingecko = myCoinGeckoAPI()
    coingecko.adapt_address_map_to_defillama()

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

initialize_tab, backtest_tab, analytics_tab, grid_tab, execution_tab = st.tabs(
    ["pool selection", "backtest", "backtest analytics", "grid analytics", "execution helper"])


def prompt_protocol_filtering() -> dict:
    result = dict()
    tvl_threshold = np.power(10, st.slider("protocol tvl threshold (log10)", min_value=0., value=6., max_value=10.,
                                           step=0.25, help="log10 of the minimum tvl to include in universe"))
    # eg merkle has no tvl
    if tvl_threshold < 2:
        tvl_threshold = -1.
    result['tvl'] = tvl_threshold
    st.write(f'protocol tvl threshold = {human_format(tvl_threshold)}')


    my_categories = ['Liquid Staking', 'Bridge', 'Lending', 'CDP', 'Dexes', 'RWA', 'Yield', 'Farm', 'Synthetics', 'Staking Pool', 'Derivatives', 'Yield Aggregator', 'Insurance', 'Liquidity manager', 'Algo-Stables', 'Decentralized Stablecoin', 'NFT Lending', 'Leveraged Farming']
    categories = st.multiselect("categories", default=my_categories,
                                options=st.session_state.all_categories,
                                help="select categories to include in universe")
    result['categories'] = categories

    # listedAt = st.slider("listedAt",
    #                      min_value=date.fromtimestamp(st.session_state.defillama.protocols['listedAt'].dropna().min()),
    #                      value=date.fromtimestamp(st.session_state.defillama.protocols['listedAt'].dropna().mean()),
    #                      max_value=date.fromtimestamp(st.session_state.defillama.protocols['listedAt'].dropna().max()),
    #                      step=timedelta(weeks=1),
    #                      help="date of listing")
    # result['listedAt'] = listedAt

    return result

def prompt_pool_filtering() -> dict:
    result = dict()
    tvlUsd = np.power(10,
                            st.slider("pool tvl threshold (log10)", min_value=0., value=5., max_value=10.,
                                      step=0.25,
                                      help="log10 of minimum tvl to include in universe"))
    if tvlUsd < 2:
        tvlUsd = -1.
    result['tvlUsd'] = tvlUsd
    st.write(f'pool tvl threshold = {human_format(tvlUsd)}')
    return result

# load whitelisted protocols from yaml
with initialize_tab:
    # instantiate defillama API
    use_oracle = st.selectbox("use_oracle", options=[False, True],
                              help="this is for depeg snipping. Please use none for now",
                              disabled=True)
    reference_asset = st.selectbox("reference_asset", options=['usd', 'eth', 'btc'],
                                   help="What asset you are investing")

    top_chains = coingecko.address_map.count().sort_values(ascending=False)[2:23].index
    chains = st.multiselect("chains", options=top_chains, default=['Arbitrum', 'Optimism', 'Ethereum'],
                            help="select chains to include in universe")
    if 'defillama' not in st.session_state:
        st.session_state.defillama = FilteredDefiLlama(reference_asset=reference_asset,
                                                       chains=chains,
                                                       oracle=coingecko,
                                                       database=parameters['input_data'][
                                                           'database'],
                                                       use_oracle=use_oracle)
        st.session_state.all_categories = list(st.session_state.defillama.protocols['category'].dropna().unique())

    whitelist, filters = st.tabs(["whitelist", "filters"])
    with whitelist:
        download_whitelist_template_button(FilteredDefiLlama.pegged_symbols[reference_asset])
        with st.form("whitelist_form"):
            if whitelist_file := st.file_uploader("Upload a set of whitelisted underlyings and protocols (download template above)", type=['xls']):
                df = pd.read_excel(whitelist_file, engine='openpyxl')
            if upload_whitelist_pressed := st.form_submit_button("Validate whitelist"):
                st.session_state.protocols = pd.read_excel(whitelist_file, sheet_name='protocols')['protocols'].tolist()
                st.session_state.underlyings = pd.read_excel(whitelist_file, sheet_name='underlyings')['underlyings'].tolist()
                st.session_state.protocol_filters = dict()
                st.session_state.upload_whitelist_pressed = upload_whitelist_pressed
    with filters:
        with st.form("filter_form"):
            underlyings_col, protocols_col = st.columns([10, 10])
            with underlyings_col:
                underlying_candidates = pd.DataFrame(
                    {'symbol': FilteredDefiLlama.pegged_symbols[reference_asset]})
                underlying_candidates['selected'] = True
                underlyings = st.data_editor(underlying_candidates[['selected', 'symbol']],
                                             use_container_width=True, hide_index=True)
            with protocols_col:
                protocol_filters = prompt_protocol_filtering()
            if validate_whitelist_pressed := st.form_submit_button("Validate underlyings and protocols"):
                st.session_state.underlyings = underlyings[underlyings['selected']]['symbol'].tolist()
                st.session_state.protocol_filters = protocol_filters
                st.session_state.validate_whitelist_pressed = validate_whitelist_pressed
                st.session_state.protocols = None
    if 'validate_whitelist_pressed' in st.session_state or 'upload_whitelist_pressed' in st.session_state:
        with st.form("pool_filter_form"):
            st.session_state.pool_filters = prompt_pool_filtering()
            if validate_pool_pressed := st.form_submit_button("Validate pool filtering"):
                st.session_state.defillama.filter(underlyings=st.session_state.underlyings,
                                                  protocol_filters=st.session_state.protocol_filters | {'selected_protocols': st.session_state.protocols},
                                                  pool_filters=st.session_state.pool_filters)
                st.write(
                    f'found {len(st.session_state.defillama.pools)} pools among {len(st.session_state.defillama.protocols)} protocols')
                st.session_state.validate_pool_pressed = validate_pool_pressed

    if ('validate_whitelist_pressed' in st.session_state or 'upload_whitelist_pressed' in st.session_state) and ('validate_pool_pressed' in st.session_state):
        with st.form("pool_selection"):
            meta_df = prettify_metadata(st.session_state.defillama.pools)
            edited_meta = st.data_editor(meta_df[['selected', 'chain', 'project', 'underlyingTokens', 'tvlUsd', 'apy',
                                                  'apyReward', 'rewardTokens', 'predictedClass', 'binnedConfidence']],
                                         use_container_width=True, hide_index=True)

            if st.form_submit_button("Predict pool yield"):
                st.session_state.defillama.pools = st.session_state.defillama.pools[edited_meta['selected']]

                with st.status(f"Fetching data from {parameters['input_data']['database']}", expanded=True) as status:
                    fetch_summary = {}
                    st.session_state.all_history: FileData = st.session_state.defillama.all_apy_history(fetch_summary=fetch_summary)
                    status.update(
                        label=f'Fetched {len([x for x in fetch_summary if ("Added" in fetch_summary[x]) or ("Created" in fetch_summary[x])])} pools \n'
                              f' Use Cache for {len([x for x in fetch_summary if fetch_summary[x] == "from db"])} pools \n '
                              f'{len([x for x in fetch_summary if fetch_summary[x] == "error"])} errors',
                        state='complete', expanded=False)

    if 'all_history' in st.session_state:
        pd.concat(st.session_state.all_history, axis=1).to_csv('all_history.csv')
        with open('all_history.csv', "rb") as file:
            st.download_button(
                label="Download all history",
                data=file,
                file_name='all_history.csv',
                mime='text/csv',
            )

with backtest_tab:
    if 'all_history' in st.session_state:
        download_grid_template_button()

        with st.form("backtest_form"):
            parameters['backtest']['end_date'] = date.today().isoformat()
            parameters['backtest']['start_date'] = (date.today() - timedelta(days=90)).isoformat()
            st.session_state.default_parameters = extract_from_paths(target=parameters, paths=parameter_keys)

            if uploaded_file := st.file_uploader("Upload a set of backtest parameters (download template above)", type=['csv']):
                st.session_state.override_grid = pd.read_csv(uploaded_file, index_col=0).to_dict(orient='records')
                # sorry hack...
                for record in st.session_state.override_grid:
                    record['label_map.apy.horizons'] = eval(record['label_map.apy.horizons'])
                    record['backtest.end_date'] = get_date_or_timedelta(record['backtest.end_date'], ref_date=date.today())
                    record['backtest.start_date'] = get_date_or_timedelta(record['backtest.start_date'], ref_date=record['backtest.end_date'])

            if st.form_submit_button("Run backtest") and ('override_grid' in st.session_state):

                if st.session_state.authentification == 'verified':
                    progress_bar1 = st.progress(value=0.0, text='Running grid...')
                    progress_bar2 = st.progress(value=0.0, text='Running backtest...')

                    st.session_state.result = VaultBacktestEngine.run_grid(parameter_grid=st.session_state.override_grid,
                                                                           parameters=parameters,
                                                                           data=st.session_state.all_history,
                                                                           progress_bar1=progress_bar1,
                                                                           progress_bar2=progress_bar2)
                    progress_bar1.progress(value=1., text='Completed grid')

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
    with grid_tab:
        st.dataframe(st.session_state.result['grid'])


with execution_tab:
    st.write("coming soon")