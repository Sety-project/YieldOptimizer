import os
from abc import abstractmethod
from copy import deepcopy, copy
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

st.title('Yield optimizer backtest \n by Sety')

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
    coingecko.address_map = coingecko.get_address_map()

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
my_categories = ['Liquid Staking', 'Bridge', 'Lending', 'CDP', 'Dexes', 'RWA', 'Yield', 'Farm', 'Synthetics',
                 'Staking Pool', 'Derivatives', 'Yield Aggregator', 'Insurance', 'Liquidity manager', 'Algo-Stables',
                 'Decentralized Stablecoin', 'NFT Lending', 'Leveraged Farming']

initialize_tab, backtest_tab, analytics_tab, grid_tab, execution_tab = st.tabs(
    ["pool selection", "backtest", "backtest analytics", "grid analytics", "execution helper"])


def prompt_protocol_filtering(all_categories) -> dict:
    result = dict()
    result['tvl'] = st.number_input("tvl threshold (in k$)", value=1000, help="minimum tvl to include in universe")*1000

    categories = st.multiselect("categories", default=my_categories,
                                options=all_categories,
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
    result['tvlUsd'] = st.number_input("tvl threshold (in k$)", value=100, help="minimum tvl to include in universe")*1000
    result['apy'] = st.number_input("apy threshold", value=2., help="minimum apy to include in universe")
    result['apyMean30d'] = st.number_input("apyMean30d threshold", value=2., help="minimum apyMean30d to include in universe")

    return result


class SessionStage:
    stage_height = {'null': 0,
                    'initialization': 1,
                    'filtering': 2,
                    'pool_selection': 3,
                    'data': 4,
                    'backtest': 5,
                    'execution': 6}

    def __init__(self):
        self.stage: int = 0
        # each stage has its own dict of properties
        # only properties up to current stage must be kept
        self.properties: list[dict] = [dict()]

    def defillama(self):
        return self.properties[-1]['defillama']

    def set(self, stage: str, **kwargs) -> None:
        print(f"stage {stage} from {self.stage}")
        properties_dict = {key: copy(value) for key, value in kwargs.items()} # not deepcopy....
        if stage == self.stage + 1:
            self.stage = stage
            self.properties.append(properties_dict)
        elif stage < self.stage:
            self.stage = stage
            self.properties = self.properties[:stage]
            self.properties[-1] = properties_dict
        elif stage == self.stage:
            self.properties[-1] = properties_dict
        else:
            raise ValueError(f"cannot move from stage {self.stage} to {stage}")


# load whitelisted protocols from yaml
with initialize_tab:
    use_oracle = st.selectbox("use_oracle", options=[False, True],
                              help="this is for depeg snipping. Please use none for now",
                              disabled=True)
    reference_asset = st.selectbox("reference_asset", options=['usd', 'eth', 'btc'],
                                   help="What asset you are investing")

    top_chains = coingecko.address_map.count().sort_values(ascending=False)[2:23].index
    chains = st.multiselect("chains", options=top_chains, default=['Arbitrum', 'Optimism', 'Ethereum'],
                            help="select chains to include in universe")
    if 'session_stage' not in st.session_state:
        st.session_state.session_stage = SessionStage()
        defillama_obj = FilteredDefiLlama(reference_asset=reference_asset,
                                      chains=chains,
                                      oracle=coingecko,
                                      database=parameters['input_data'][
                                          'database'],
                                      use_oracle=use_oracle)
        st.session_state.session_stage.set(1,
                                           defillama=defillama_obj)

    if st.session_state.session_stage.stage >= 1:
        filters, whitelist = st.tabs(["filters", "whitelist"])
        with filters:
            with st.form("filter_form"):
                underlyings_col, protocols_col, pool_col = st.columns([20, 20, 20])
                with underlyings_col:
                    st.write("### Underlying filter")
                    underlying_candidates = pd.DataFrame(
                        {'symbol': FilteredDefiLlama.pegged_symbols[reference_asset]})
                    underlying_candidates['selected'] = True
                    underlyings = st.data_editor(underlying_candidates[['selected', 'symbol']],
                                                 use_container_width=True, hide_index=True)
                    underlyings = underlyings[underlyings['selected']]['symbol'].tolist()
                with protocols_col:
                    st.write("### Protocol filter")
                    protocol_filters = prompt_protocol_filtering(all_categories=list(
                        st.session_state.session_stage.defillama().protocols['category'].dropna().unique()))
                with pool_col:
                    st.write("### Pool filter")
                    pool_filters = prompt_pool_filtering()

                filter_form_pressed = st.form_submit_button("Validate underlyings and protocols")

        with whitelist:
            download_whitelist_template_button(FilteredDefiLlama.pegged_symbols[reference_asset])
            with st.form("whitelist_form"):
                upload_col, pool_col = st.columns([20, 20])
                with upload_col:
                    st.write("### Upload")
                    whitelist_file = st.file_uploader(
                            "Upload a set of whitelisted underlyings and protocols (download template above)",
                            type=['xls'])
                with pool_col:
                    st.write("### Pool filter")
                    pool_filters = prompt_pool_filtering()

                if st.form_submit_button("Validate whitelist"):
                    if whitelist_file is None:
                        st.write("Please upload a whitelist")
                        filter_form_pressed = False
                    else:
                        underlyings = pd.read_excel(whitelist_file, sheet_name='underlyings')[
                            'underlyings'].tolist()
                        protocol_filters = {
                            'selected_protocols': pd.read_excel(whitelist_file, sheet_name='protocols')[
                                'protocols']}
                        filter_form_pressed = True

        if filter_form_pressed and st.session_state.session_stage.stage == 1:
            st.session_state.session_stage.set(2, defillama=st.session_state.session_stage.defillama())
            st.session_state.session_stage.defillama().filter(
                underlyings=underlyings,
                protocol_filters=protocol_filters,
                pool_filters=pool_filters)
            st.write(
                f"found {len(st.session_state.session_stage.defillama().pools)} pools among {len(st.session_state.session_stage.defillama().protocols)} protocols")
    if st.session_state.session_stage.stage >= 2:
        with st.form("pool_selection"):
            meta_df = prettify_metadata(st.session_state.session_stage.defillama().pools)
            edited_meta = st.data_editor(meta_df[['selected', 'chain', 'project', 'underlyingTokens', 'tvlUsd', 'apy',
                                                  'apyReward', 'rewardTokens', 'predictedClass', 'binnedConfidence']],
                                         use_container_width=True, hide_index=True)

            #st.checkbox("Refresh DB", key='resfresh_db', value=False)
            if st.form_submit_button("Predict pool yield"):
                st.session_state.session_stage.set(3, defillama=st.session_state.session_stage.defillama())
                st.session_state.session_stage.defillama().pools = st.session_state.session_stage.defillama().pools[edited_meta['selected']]

                with st.spinner(f"Fetching data from {parameters['input_data']['database']}"):
                    fetch_summary = {}
                    st.session_state.session_stage.set(4,
                                                       defillama=st.session_state.session_stage.defillama(),
                                                       all_history=st.session_state.session_stage.defillama().refresh_apy_history(fetch_summary=fetch_summary))
                    errors = len([x for x in fetch_summary if fetch_summary[x] == "error"])
                    st.write(f'Fetched {len([x for x in fetch_summary if ("Added" in fetch_summary[x]) or ("Created" in fetch_summary[x])])} pools \n'
                              f' Use Cache for {len([x for x in fetch_summary if fetch_summary[x] == "from db"])} pools \n '
                              f'{errors} errors{ ("excluding errorenous pools unless you re-fetch (usually a DefiLama API glitch") if errors > 0 else ""}')

    if st.session_state.session_stage.stage >= 4:
        pd.concat(st.session_state.session_stage.properties[-1]['all_history'], axis=1).to_csv('all_history.csv')
        with open('all_history.csv', "rb") as file:
            st.download_button(
                label="Download all history",
                data=file,
                file_name='all_history.csv',
                mime='text/csv',
            )

with backtest_tab:
    if st.session_state.session_stage.stage >= 4:
        download_grid_template_button()

        with st.form("backtest_form"):
            parameters['backtest']['end_date'] = date.today().isoformat()
            parameters['backtest']['start_date'] = (date.today() - timedelta(days=90)).isoformat()
            default_parameters = extract_from_paths(target=parameters, paths=parameter_keys)

            if uploaded_file := st.file_uploader("Upload a set of backtest parameters (download template above)", type=['csv']):
                override_grid = pd.read_csv(uploaded_file, index_col=0).to_dict(orient='records')
                # sorry hack...
                for record in override_grid:
                    record['label_map.apy.horizons'] = eval(record['label_map.apy.horizons'])
                    record['backtest.end_date'] = get_date_or_timedelta(record['backtest.end_date'], ref_date=date.today())
                    record['backtest.start_date'] = get_date_or_timedelta(record['backtest.start_date'], ref_date=record['backtest.end_date'])

            if st.form_submit_button("Run backtest") and ('override_grid' in st.session_state):
                if st.session_state.authentification == 'verified':
                    progress_bar1 = st.progress(value=0.0, text='Running grid...')
                    progress_bar2 = st.progress(value=0.0, text='Running backtest...')

                    st.session_state.session_stage.set(5, result=VaultBacktestEngine.run_grid(parameter_grid=override_grid,
                                                                           parameters=parameters,
                                                                           data=st.session_state.session_stages.properties[-1]['all_history'],
                                                                           progress_bar1=progress_bar1,
                                                                           progress_bar2=progress_bar2))
                    progress_bar1.progress(value=1., text='Completed grid')
                else:
                    with st.sidebar.expander("Expand me"):
                        st.session_state.user_tg_handle = st.sidebar.text_input("Enter your tg handle to backtest:")
                        if st.session_state.authentification != "verified":
                            check_whitelist()


with analytics_tab:
    if (st.session_state.session_stage.stage >= 5) and (selected_run := st.selectbox('Select run name', st.session_state.session_stage.properties[-1]['result']['runs'].keys())):
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
    if (st.session_state.session_stage.stage >= 5):
        st.dataframe(st.session_state.session_stage.properties[-1]['result']['grid'])


with execution_tab:
    st.write("coming soon")