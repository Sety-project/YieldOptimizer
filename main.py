import os
from copy import copy
from datetime import timedelta, date

import pandas as pd
import streamlit as st
import yaml

from scrappers.defillama_history.defillama import FilteredDefiLlama
from strategies.vault_backtest import VaultBacktestEngine
from utils.io_utils import extract_from_paths, get_date_or_timedelta
from utils.streamlit_utils import download_grid_template_button, \
    display_single_backtest, download_whitelist_template_button, MyProgressBar, prettify_metadata, \
    authentification_sidebar, coingecko, prompt_initialization, prompt_protocol_filtering, prompt_pool_filtering, \
    check_whitelist

pd.options.mode.chained_assignment = None

st.title('Yield optimizer backtest \n by Sety')

authentification_sidebar()

# load default parameters from yaml
with open(os.path.join(os.sep, os.getcwd(), "config", 'params.yaml'), 'r') as fp:
    parameters = yaml.safe_load(fp)

initialize_tab, backtest_tab, analytics_tab, grid_tab, execution_tab = st.tabs(
    ["pool selection", "backtest", "backtest analytics", "grid analytics", "execution helper"])

if 'stage' not in st.session_state:
    st.session_state.stage = 0

# load whitelisted protocols from yaml
with initialize_tab:
    use_oracle, reference_asset, chains = prompt_initialization()
    if initialize_pressed := st.button("Initialize"):
        st.session_state.defillama = FilteredDefiLlama(reference_asset=reference_asset,
                                                       chains=chains,
                                                       oracle=coingecko,
                                                       database=parameters['input_data'][
                                                           'database'],
                                                       use_oracle=use_oracle)
        st.session_state.all_categories = list(st.session_state.defillama.protocols['category'].dropna().unique())
        st.session_state.stage = 1
        initialize_pressed = False

    if st.session_state.stage >= 1:
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
                    protocol_filters = prompt_protocol_filtering(all_categories=st.session_state.all_categories)
                with pool_col:
                    st.write("### Pool filter")
                    pool_filters = prompt_pool_filtering()

                if validate_form_pressed := st.form_submit_button("Validate underlyings and protocols"):
                    st.session_state.defillama.filter(
                        underlyings=underlyings,
                        protocol_filters=protocol_filters,
                        pool_filters=pool_filters)
                    st.write(
                        f"found {len(st.session_state.defillama.pools)} pools among {len(st.session_state.defillama.protocols)} protocols")
                    st.session_state.stage = 2

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

                if whitelist_form_pressed := st.form_submit_button("Validate whitelist"):
                    if whitelist_file is None:
                        st.write("Please upload a whitelist")
                        whitelist_form_pressed = False
                    else:
                        underlyings = pd.read_excel(whitelist_file, sheet_name='underlyings')[
                            'underlyings'].tolist()
                        protocol_filters = {
                            'selected_protocols': pd.read_excel(whitelist_file, sheet_name='protocols')[
                                'protocols'].unique()}
                        st.session_state.defillama.filter(
                            underlyings=underlyings,
                            protocol_filters=protocol_filters,
                            pool_filters=pool_filters)
                        st.write(
                            f"found {len(st.session_state.defillama.pools)} pools among {len(st.session_state.defillama.protocols)} protocols")
                        st.session_state.stage = 2

    if st.session_state.stage >= 2:
        with st.form("pool_selection"):
            meta_df = prettify_metadata(st.session_state.defillama.pools)
            edited_meta = st.data_editor(meta_df[['selected', 'chain', 'project', 'underlyingTokens', 'tvlUsd', 'apy',
                                                  'apyReward', 'rewardTokens', 'predictedClass', 'binnedConfidence']],
                                         use_container_width=True, hide_index=True)

            #st.checkbox("Refresh DB", key='resfresh_db', value=False)
            if st.form_submit_button("Predict pool yield"):
                st.session_state.defillama.pools = st.session_state.defillama.pools[edited_meta['selected']]
                st.session_state.stage = 3

                progress_bar = MyProgressBar(value=0.0,
                                             length=st.session_state.defillama.pools.shape[0],
                                             text=f"Fetching data from {parameters['input_data']['database']}")
                fetch_summary = {}
                st.session_state.all_history = st.session_state.defillama.refresh_apy_history(fetch_summary=fetch_summary, progress_bar=progress_bar)
                st.session_state.stage = 4
                errors = len([x for x in fetch_summary if "error" in fetch_summary[x][0]])
                progress_bar.progress_bar.progress(value=1.0, text=f'Fetched {len([x for x in fetch_summary if ("Added" in fetch_summary[x][0]) or ("Created" in fetch_summary[x][0])])} pools \n'
                          f' Use Cache for {len([x for x in fetch_summary if "from db" in fetch_summary[x][0]])} pools \n '
                          f'{errors} errors{ ("excluding errorenous pools unless you re-fetch (usually a DefiLama API glitch") if errors > 0 else ""}')

    if st.session_state.stage >= 4:
        pd.concat(st.session_state.all_history, axis=1).to_csv('all_history.csv')
        with open('all_history.csv', "rb") as file:
            st.download_button(
                label="Download all history",
                data=file,
                file_name='all_history.csv',
                mime='text/csv',
            )

with backtest_tab:
    if st.session_state.stage >= 4:
        download_grid_template_button()

        with st.form("backtest_form"):
            parameters['backtest']['end_date'] = date.today().isoformat()
            parameters['backtest']['start_date'] = (date.today() - timedelta(days=90)).isoformat()
            default_parameters = extract_from_paths(target=parameters,
                                                    paths=['strategy.initial_wealth',
                                                           'run_parameters.models.apy.TrivialEwmPredictor.params.cap',
                                                           'run_parameters.models.apy.TrivialEwmPredictor.params.halflife',
                                                           'strategy.cost',
                                                           'strategy.gas',
                                                           'strategy.base_buffer',
                                                           "run_parameters.models.apy.TrivialEwmPredictor.params.horizon",
                                                           "label_map.apy.horizons",
                                                           "strategy.concentration_limit",
                                                           "backtest.end_date",
                                                           "backtest.start_date"])

            override_grid = None
            if uploaded_file := st.file_uploader("Upload a set of backtest parameters (download template above)", type=['csv']):
                override_grid = pd.read_csv(uploaded_file, index_col=0).to_dict(orient='records')
                # sorry hack...
                for record in override_grid:
                    record['label_map.apy.horizons'] = eval(record['label_map.apy.horizons'])
                    record['backtest.end_date'] = get_date_or_timedelta(record['backtest.end_date'], ref_date=date.today())
                    record['backtest.start_date'] = get_date_or_timedelta(record['backtest.start_date'], ref_date=record['backtest.end_date'])

            if st.form_submit_button("Run backtest") and override_grid is not None:
                if st.session_state.authentification == 'verified':
                    progress_bar1 = st.progress(value=0.0, text='Running grid...')
                    progress_bar2 = st.progress(value=0.0, text='Running backtest...')

                    st.session_state.result = VaultBacktestEngine.run_grid(parameter_grid=override_grid,
                                                                           parameters=parameters,
                                                                           data=
                                                                           st.session_state.all_history,
                                                                           progress_bar1=progress_bar1,
                                                                           progress_bar2=progress_bar2)
                    progress_bar1.progress(value=1., text='Completed grid')
                    st.session_state.stage = 5
                else:
                    with st.sidebar.expander("Expand me"):
                        st.session_state.user_tg_handle = st.sidebar.text_input("Enter your tg handle to backtest:")
                        if st.session_state.authentification != "verified":
                            check_whitelist()

with analytics_tab:
    if (st.session_state.stage >= 5) and (selected_run := st.selectbox('Select run name', st.session_state.result['runs'].keys())):
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
    if (st.session_state.stage >= 5):
        st.dataframe(st.session_state.result['grid'])

with execution_tab:
    st.write("coming soon")