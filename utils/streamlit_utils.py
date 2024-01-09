import html
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from plotly import express as px


def check_whitelist():
    if st.session_state.user_tg_handle in st.secrets.whitelist:
        st.session_state.authentification = "verified"
        st.sidebar.success(f'logged in as {st.session_state.user_tg_handle}')
    else:
        st.session_state.authentification = "incorrect"
        st.warning(
            html.unescape(
                'chat https://t.me/daviddarr, then enter your tg handle in the sidebar to get access'
            )
        )
    st.session_state.password = ""


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def download_grid_template_button() -> None:
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

def download_whitelist_template_button(underlyings_candidates: list[str]) -> None:
    # button to download grid template
    underlyings = pd.DataFrame({'underlyings': underlyings_candidates})
    protocols = pd.DataFrame(
        {'protocols': ['curve-dex',
                       'balancer',
                       'pancakeswap-amm',
                       'venus-isolated-pools',
                       'lido',
                       'aave-v2',
                       'morpho-aavev3',
                       'uniswap-v2',
                       'thena-v1',
                       'morpho-compoundcompound-v3',
                       'morpho-aave',
                       'curve-finance',
                       'convex-finance',
                       'compound',
                       'frax-ether',
                       'aura',
                       'venus-core-pool',
                       'aave-v3',
                       'uniswap-v3',
                       'stargate',
                       'makerdao',
                       'balancer-v2',
                       'rocket-pool',
                       'pancakeswap-amm-v3',
                       'spark'
                       ]})
    with pd.ExcelWriter('whitelist_template.xls', engine='xlsxwriter', mode='w') as writer:
        underlyings.to_excel(writer, sheet_name='underlyings', index=False)
        protocols.to_excel(writer, sheet_name='protocols', index=False)
    with open('whitelist_template.xls', "rb") as download_file:
        st.download_button(
            label="Download backtest grid template",
            data=download_file,
            file_name='whitelist_template.xls',
            mime='text/xls',
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


def show_edits() -> list:
    return [{'strategy.initial_wealth': np.power(10,
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
