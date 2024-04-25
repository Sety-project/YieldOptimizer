import os
import threading
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yaml
from plotly import express as px

from scrappers.defillama_history.coingecko import myCoinGeckoAPI
from utils.telegram_bot import check_whitelist

coingecko = myCoinGeckoAPI()
with st.spinner('fetching meta data'):
    coingecko.address_map = coingecko.get_address_map()


def authentification_sidebar() -> None:
    if 'skip_authentification' in st.secrets:
        st.session_state.authentification = "verified"
        return

    if 'authentification' not in st.session_state:
        st.session_state.authentification = "unverified"

    if st.sidebar.text_input("Enter your tg handle to backtest:", key='user_tg_handle'):
        if st.session_state.authentification != "verified" and check_whitelist(st.session_state.user_tg_handle):
            st.session_state.authentification = "verified"

        if st.session_state.user_tg_handle in st.secrets.admins:
            st.sidebar.dataframe(
                st.session_state.database.read_sql_query(f'''SELECT * FROM interactions ORDER BY timestamp ASC;'''))
            if st.sidebar.text_input("DB to reset (don't !)", key='db_delete'):
                st.session_state.database.delete()
                st.caption(f"deleted {len(st.session_state.database.list_tables())} from {st.session_state.db_delete}")


def load_parameters() -> dict:
    if parameter_file := st.sidebar.file_uploader("upload parameters", type=['yaml']):
        result = yaml.safe_load(parameter_file)
    elif 'parameters' not in st.session_state:
        with open(os.path.join(os.sep, os.getcwd(), "config", 'params.yaml'), 'r') as fp:
            result = yaml.safe_load(fp)
    else:
        result = st.session_state.parameters

    return result


def parameter_override():
    st.sidebar.subheader("Session parameters", divider='grey')
    st.sidebar.json(st.session_state.parameters)
    # parameters_override = st.sidebar.data_editor(st.session_state.parameters, use_container_width=True, height=1000, key='parameters')


def prompt_initialization():
    def reset():
        st.session_state.stage = 0
    use_oracle = st.selectbox("pegged asset price fluctuation as APY", options=[False, True],
                              help="this is for depeg snipping. Please use none for now",
                              disabled=True, on_change=reset)
    reference_asset = st.selectbox("reference_asset", options=['usd', 'eth', 'btc'],
                                   help="What asset you are investing", on_change=reset)

    top_chains = coingecko.address_map.count().sort_values(ascending=False).index[2:]
    chains = st.multiselect("chains", options=top_chains, default=['Arbitrum', 'Optimism', 'Base', 'Polygon', 'Ethereum'],
                            help="select chains to include", on_change=reset)

    return use_oracle, reference_asset, chains


def prompt_protocol_filtering(all_categories) -> dict:
    result = {}
    result['tvl'] = st.number_input("tvl threshold (in k$)", value=1000, help="minimum tvl to include in universe")*1000

    categories = st.multiselect("categories",
                                default=None,
                                placeholder="all",
                                options=all_categories,
                                help="select categories to include in universe")
    result['categories'] = categories or all_categories

    # listedAt = st.slider("listedAt",
    #                      min_value=date.fromtimestamp(st.session_state.defillama.protocols['listedAt'].dropna().min()),
    #                      value=date.fromtimestamp(st.session_state.defillama.protocols['listedAt'].dropna().mean()),
    #                      max_value=date.fromtimestamp(st.session_state.defillama.protocols['listedAt'].dropna().max()),
    #                      step=timedelta(weeks=1),
    #                      help="date of listing")
    # result['listedAt'] = listedAt

    return result


def prompt_pool_filtering(reference_asset: str=None) -> dict:
    defaults_map = {'usd': {'apy': 3.0, 'apyMean30d': 3.0, 'tvlUsd': 1000},
                    'eth': {'apy': 0.0, 'apyMean30d': 0.0, 'tvlUsd': 1000},
                    'btc': {'apy': 0.0, 'apyMean30d': 0.0, 'tvlUsd': 1000}}
    result = {}
    # TODO: bad bug in streamlit, can't use defaults_map[reference_asset] as default value -> "DuplicateWidgetID: There are multiple identical st.number_input widgets with the same generated key."
    # result['tvlUsd'] = st.number_input("tvl threshold (in k$)", value=defaults_map[reference_asset]['tvlUsd'], help="minimum tvl to include in universe")*1000
    # result['apy'] = st.number_input("apy threshold", value=defaults_map[reference_asset]['apy'], help="minimum apy to include in universe")
    # result['apyMean30d'] = st.number_input("apyMean30d threshold", value=defaults_map[reference_asset]['apyMean30d'], help="minimum apyMean30d to include in universe")
    result['tvlUsd'] = st.number_input("tvl threshold (in k$)", value=100., help="minimum tvl to include in universe")*1000
    result['apy'] = st.number_input("apy threshold", value=0., help="minimum apy to include in universe")
    result['apyMean30d'] = st.number_input("apyMean30d threshold", value=0., help="minimum apyMean30d to include in universe")

    return result


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


def download_grid_button() -> None:
    with open(os.path.join(os.sep, os.getcwd(), "config", 'grid.yaml'), "r") as download_file:
        st.download_button(
            label="Download backtest grid template",
            data=download_file,
            file_name='grid.yaml',
            mime='text/yaml',
        )


def download_whitelist_template_button(underlyings_candidates: list[str]) -> None:
    # button to download grid template
    with open(os.path.join(os.sep, os.getcwd(), "config", 'whitelist.yaml'), "r") as download_file:
        st.download_button(
            label="Download backtest grid template",
            data=download_file,
            file_name='whitelist_template.yaml',
            mime='text/yaml',
        )


def display_single_backtest(run_idx: str) -> None:
    height = 1000
    width = 1500
    backtest = st.session_state.result['runs'][run_idx]
    params = st.session_state.result['grid'].iloc[list(st.session_state.result['runs'].keys()).index(run_idx)]
    # plot_perf(backtest, override_grid['strategy.base_buffer'][0], height=height, width=width)
    apy = (backtest['apy'] * backtest['weights'].divide(backtest['weights'].sum(axis=1), axis=0)).drop(
        columns='total').reset_index().melt(id_vars='index', var_name='pool', value_name='apy').dropna()
    apyReward = (backtest['apyReward'] * backtest['weights'].divide(backtest['weights'].sum(axis=1),
                                                                            axis=0)).drop(
        columns='total').reset_index().melt(id_vars='index', var_name='pool', value_name='apy').dropna()
    apyReward['pool'] = apyReward['pool'].apply(lambda x: f"reward_{x}")
    apy['apy'] = apy['apy'] - apyReward['apy']
    apy = pd.concat([apy, apyReward], axis=0)
    apy['apy'] = apy['apy'] * 100
    avg_apy = apy.groupby('pool').mean()
    truncated_avg_apy = avg_apy[avg_apy['apy'] > -9999].reset_index()

    # show avg over backtest
    details = dict()
    year_fraction = (backtest.index.max() - backtest.index.min()).total_seconds()/24/3600/365.25
    details['gas_apy'] = -100 * backtest[('pnl', 'gas')].sum() / backtest[('pnl', 'wealth')].mean() / year_fraction
    details['slippage_apy'] = -100 * backtest[('pnl', 'tx_cost')].sum() / backtest[('pnl', 'wealth')].mean() / year_fraction
    details['nb trade per d'] = backtest[('pnl', 'gas')].sum() /  params['strategy.gas'] / year_fraction / 365.25
    details['churn (nb of days to trade 100% capital)'] = 365.25 * year_fraction /(-details['slippage_apy']/100 / params['strategy.cost'])

    st.subheader(f'total apy = {avg_apy["apy"].sum()/100:.1%}')
    st.table(details)
    st.plotly_chart(
        px.pie(truncated_avg_apy, names='pool', values='apy', title='apy * weights (%)', height=height / 2,
               width=width / 2))

    # show all apy drivers over time
    truncated_apy = apy[apy['pool'].isin(truncated_avg_apy['pool'])]
    st.plotly_chart(
        px.bar(truncated_apy, x='index', y='apy', color='pool', title='apy (%)', height=height, width=width, barmode='stack'))

    ## show all weights over time
    weights = backtest['weights'].drop(columns='total').reset_index().melt(id_vars='index', var_name='pool',
                                                                               value_name='weights').dropna()
    truncated_weights = weights[weights['pool'].isin(truncated_avg_apy['pool'])]
    st.plotly_chart(
        px.bar(truncated_weights, x='index', y='weights', color='pool', title='Allocations ($)', height=height,
               width=width, barmode='stack'))

    # show all dilutors over time
    dilutor = backtest['dilutor'].drop(columns='total').reset_index().melt(id_vars='index', var_name='pool',
                                                                               value_name='dilutor').dropna()
    dilutor['dilutor'] = 100 * (1 - dilutor['dilutor'])
    truncated_dilutor = dilutor[dilutor['pool'].isin(truncated_avg_apy['pool'])]
    st.plotly_chart(
        px.line(truncated_dilutor, x='index', y='dilutor', color='pool', title='pool ownership (%)', height=height,
                width=width))


def display_heatmap(grid: pd.DataFrame, metrics, ind, col, filtering):
    fig = plt.figure()

    # filter
    filtered_grid = grid[np.logical_and.reduce([
        grid[filter_c].explode() == filter_v
        for filter_c, filter_v in filtering.items()])]

    # pivot and display
    for i, values in enumerate(metrics):
        for j, column in enumerate(col):
            df = filtered_grid.pivot_table(values=values, index=ind, columns=col) * 100
            ax = fig.add_subplot(len(metrics), 1, i + j + 1)
            ax.set_title(f'{values} by {ind[0]}, {column}')
            sns.heatmap(data=df, ax=ax)#, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': 8})

    return fig


class MyProgressBar:
    '''A progress bar with increment progress (why did i have to do that...)'''
    def __init__(self, length, **kwargs):
        self.progress_bar = st.progress(**kwargs)
        self._lock = threading.Lock()
        self.length: int = length
        self.progress_idx: int = 0

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

    def increment(self, value: int = 1, text: str = ''):
        assert 0 <= self.progress_idx <= self.length
        with self:
            self.progress_idx += value
            self.progress_bar.progress(value=np.clip(self.progress_idx / self.length, a_min=0, a_max=1), text=text)


