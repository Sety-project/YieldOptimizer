import os
import threading
from copy import deepcopy

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from plotly import express as px

from scrappers.defillama_history.coingecko import myCoinGeckoAPI
from utils.telegram_bot import check_whitelist

coingecko = myCoinGeckoAPI()
with st.spinner('fetching meta data'):
    coingecko.address_map = coingecko.get_address_map()


def authentification_sidebar():
    if 'authentification' not in st.session_state:
        st.session_state.authentification = "unverified"
    if st.sidebar.text_input("Enter your tg handle to backtest:", key='user_tg_handle'):
        if st.session_state.authentification != "verified":
            check_whitelist(st.session_state.user_tg_handle)
            st.session_state.authentification = "verified"

        if st.session_state.user_tg_handle in st.secrets.admins and st.sidebar.text_input("DB to reset (don't !)", key='db_delete'):
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
    use_oracle = st.selectbox("use_oracle", options=[False, True],
                              help="this is for depeg snipping. Please use none for now",
                              disabled=True, on_change=reset)
    reference_asset = st.selectbox("reference_asset", options=['usd', 'eth', 'btc'],
                                   help="What asset you are investing", on_change=reset)
    top_chains = coingecko.address_map.count().sort_values(ascending=False)[2:23].index
    chains = st.multiselect("chains", options=top_chains, default=['Arbitrum', 'Optimism', 'Ethereum'],
                            help="select chains to include in universe", on_change=reset)

    return use_oracle, reference_asset, chains


def prompt_protocol_filtering(all_categories,
                              my_categories=['Liquid Staking', 'Bridge', 'Lending', 'CDP', 'Dexes', 'RWA', 'Yield',
                                             'Farm', 'Synthetics',
                                             'Staking Pool', 'Derivatives', 'Yield Aggregator', 'Insurance',
                                             'Liquidity manager', 'Algo-Stables',
                                             'Decentralized Stablecoin', 'NFT Lending', 'Leveraged Farming']) -> dict:
    def reset():
        st.session_state.stage = 1
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
    def reset():
        st.session_state.stage = 2
    result = dict()
    result['tvlUsd'] = st.number_input("tvl threshold (in k$)", value=100, help="minimum tvl to include in universe")*1000
    result['apy'] = st.number_input("apy threshold", value=2., help="minimum apy to include in universe")
    result['apyMean30d'] = st.number_input("apyMean30d threshold", value=2., help="minimum apyMean30d to include in universe")

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


def display_single_backtest(backtest: pd.DataFrame) -> None:
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
    truncated_avg_apy = avg_apy[avg_apy['apy'] > -1].reset_index()
    # show avg over backtest
    st.write(f'total apy = {avg_apy["apy"].sum()/100:.1%}')
    st.plotly_chart(
        px.pie(truncated_avg_apy, names='pool', values='apy', title='apy * weights (%)', height=height / 2,
               width=width / 2))
    # show all apy drivers over time
    truncated_apy = apy[apy['pool'].isin(truncated_avg_apy['pool'])]
    st.plotly_chart(
        px.bar(truncated_apy, x='index', y='apy', color='pool', title='apy (%)', height=height, width=width, barmode='stack'))
    # show all weights over time
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
        px.line(truncated_dilutor, x='index', y='dilutor', color='pool', title='alloc / tvl (%)', height=height,
                width=width))


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


class MyProgressBar:
    '''A progress bar with increment progress (why did i have to do that...)'''
    def __init__(self, length, **kwargs):
        self.progress_bar = st.progress(**kwargs)
        self._lock = threading.Lock()
        self.length: float = length
        self._progress = 0

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

    def increment(self, value: float=1., text: str = ''):
        assert value >= 0
        assert value <= self.length
        with self:
            self._progress += value/self.length
            self.progress_bar.progress(value=np.clip(self._progress, a_min=0, a_max=1), text=text)


