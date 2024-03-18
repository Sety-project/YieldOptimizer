import html
import os
import sys
from datetime import timedelta, date

import matplotlib
import pandas as pd
import streamlit as st
import yaml

from scrappers.defillama_history.defillama import FilteredDefiLlama
from strategies.vault_backtest import VaultBacktestEngine
from utils.io_utils import extract_from_paths, get_date_or_timedelta
from utils.postgres import SqlApi
from utils.streamlit_utils import download_grid_button, \
    display_single_backtest, download_whitelist_template_button, MyProgressBar, prettify_metadata, \
    authentification_sidebar, coingecko, prompt_initialization, prompt_protocol_filtering, prompt_pool_filtering, \
    load_parameters, parameter_override, display_heatmap

assert (sys.version_info >= (3, 10)), "Please use Python 3.10 or higher"

pd.options.mode.chained_assignment = None

st.title('Yield optimizer backtest \n by Sety')

st.session_state.parameters = load_parameters()
st.session_state.database = SqlApi(name=st.session_state.parameters['input_data']['database'],
                  pool_size=st.session_state.parameters['input_data']['async']['pool_size'],
                  max_overflow=st.session_state.parameters['input_data']['async']['max_overflow'],
                  pool_recycle=st.session_state.parameters['input_data']['async']['pool_recycle'])
authentification_sidebar()
parameter_override()

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
                                                       database=st.session_state.database,
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
                    st.subheader("Underlying filter", divider='grey')
                    underlying_candidates = pd.DataFrame(
                        {'symbol': FilteredDefiLlama.pegged_symbols[reference_asset]})
                    underlying_candidates['selected'] = True
                    underlyings = st.data_editor(underlying_candidates[['selected', 'symbol']],
                                                 use_container_width=True, hide_index=True)
                    underlyings = underlyings[underlyings['selected']]['symbol'].tolist()
                with protocols_col:
                    st.subheader("Protocol filter", divider='grey')
                    protocol_filters = prompt_protocol_filtering(all_categories=st.session_state.all_categories)
                with pool_col:
                    st.subheader("Pool filter", divider='grey')
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
                    st.subheader("Upload", divider='grey')
                    whitelist_file = st.file_uploader(
                            "Upload a set of whitelisted underlyings and protocols (download template above)",
                            type=['yaml'], key='whitelist_file')
                with pool_col:
                    st.subheader("Pool filter", divider='grey')
                    pool_filters = prompt_pool_filtering()

                if whitelist_form_pressed := st.form_submit_button("Validate whitelist"):
                    if whitelist_file is None:
                        st.write("Please upload a whitelist")
                        whitelist_form_pressed = False
                    else:
                        whitelist = yaml.safe_load(whitelist_file)
                        st.session_state.defillama.filter(
                            underlyings=whitelist['underlyings'],
                            protocol_filters={'selected_protocols': whitelist['protocols']},
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

            populate_db = st.checkbox("Populate DB (slow !)", value=False)
            if st.form_submit_button("Predict pool yield"):
                st.session_state.defillama.pools = st.session_state.defillama.pools[edited_meta['selected']]
                st.session_state.stage = 3

                progress_bar = MyProgressBar(value=0.0,
                                             length=st.session_state.defillama.pools.shape[0],
                                             text=f"Fetching data from {st.session_state.parameters['input_data']['database']}")
                fetch_summary = {}
                st.session_state.all_history = st.session_state.defillama.refresh_apy_history(fetch_summary=fetch_summary, progress_bar=progress_bar, populate_db=populate_db)
                st.session_state.stage = 4
                errors = len([x for x in fetch_summary if "error" in fetch_summary[x][0]])
                progress_bar.progress_bar.progress(value=1.0, text=f'Fetched {len([x for x in fetch_summary if ("Added" in fetch_summary[x][0]) or ("Created" in fetch_summary[x][0])])} pools \n'
                          f' Use Cache for {len([x for x in fetch_summary if "from db" in fetch_summary[x][0]])} pools \n '
                          f'{errors} errors{ (", excluding errorenous pools unless you re-fetch (usually a DefiLama API glitch") if errors > 0 else ""}')

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
        download_grid_button()

        with st.form("backtest_form"):
            st.session_state.parameters['backtest']['end_date'] = date.today().isoformat()
            st.session_state.parameters['backtest']['start_date'] = (date.today() - timedelta(days=90)).isoformat()
            default_parameters = extract_from_paths(target=st.session_state.parameters,
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

            if uploaded_file := st.file_uploader("Upload a set of backtest parameters (download template above)", type=['yaml']):
                override_grid = yaml.safe_load(uploaded_file)
            else:
                with open(os.path.join(os.sep, os.getcwd(), "config", 'grid.yaml'), 'r') as uploaded_file:
                    override_grid = yaml.safe_load(uploaded_file)
            # relative date support
            for record in override_grid:
                record['backtest.end_date'] = get_date_or_timedelta(record['backtest.end_date'],
                                                                    ref_date=date.today())
                record['backtest.start_date'] = get_date_or_timedelta(record['backtest.start_date'],
                                                                      ref_date=record['backtest.end_date'])

            if st.form_submit_button("Run backtest") and override_grid is not None:
                st.session_state.result = VaultBacktestEngine.run_grid(parameter_grid=override_grid,
                                                                       parameters=st.session_state.parameters,
                                                                       data=
                                                                       st.session_state.all_history)
                st.session_state.stage = 5
        st.subheader("Backtest grid", divider='grey')
        st.json(override_grid)

with analytics_tab:
    if (st.session_state.stage >= 5) and (selected_run := st.selectbox('Select run name', st.session_state.result['runs'].keys())):
        if st.session_state.authentification == 'verified':
            display_single_backtest(selected_run)
            pd.concat(st.session_state.result['runs'], axis=1).to_csv('logs/runs.csv')
            with open('logs/runs.csv', "rb") as file:
                st.download_button(
                    label="Download runs",
                    data=file,
                    file_name='runs.csv',
                    mime='text/csv',
                )
        else:
            st.warning(
                html.unescape(
                    'chat https://t.me/Pronoia_Bot, then enter your tg handle in the sidebar to get access'
                )
            )

with grid_tab:
    if st.session_state.stage >= 5 and st.session_state.authentification == 'verified':
        grid = st.session_state.result['grid']
        fields = grid.columns
        metric_fields = ['perf', 'tx_cost', 'avg_entropy']

        st.dataframe(grid)
        with st.form("heatmap_form"):
            X_options = [ind for ind in fields if ind not in metric_fields]
            X = st.selectbox('X axis', X_options, index=list(X_options).index("strategy.initial_wealth"))

            Y_options = [col for col in fields if col not in [X] + metric_fields]
            Y = st.selectbox('Y axis', Y_options, index=list(Y_options).index("strategy.gas"))

            metrics = st.selectbox('Metrics', metric_fields, index=0)

            filtering_options = [field for field in fields if field not in [X, Y] + metric_fields]
            filtering = {
                field: st.selectbox(
                    field,
                    list(
                        (
                            grid[field].explode()
                            if type(grid[field].iloc[0]) is list
                            else grid[field]
                        ).unique()
                    ),
                    index=0,
                )
                for field in filtering_options
            }
            st.write("Heatmap of the grid search results. The color intensity is:\n "
                     "- proportional to the metric value\n"
                     "- for values set in the filters"
                     "- averaged over the empty fields in filters\n")
            if st.form_submit_button("Display heatmap"):
                st.pyplot(display_heatmap(grid, metrics=[metrics], ind=[X], col=[Y], filtering=filtering),
                          use_container_width=True)

with execution_tab:
    st.write("coming soon")