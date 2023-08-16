import os
from typing import Union
from datetime import timedelta
import numpy as np
import pandas as pd
from strategies.vault_rebalancing import VaultRebalancingStrategy
from strategies.research_engine import ResearchEngine

class VaultBacktestEngine:
    def __init__(self, params: dict):
        self.parameters = params
        self.parameters['start_date'] = pd.to_datetime(self.parameters['start_date'], unit='ns', utc=True)
        self.parameters['end_date'] = pd.to_datetime(self.parameters['end_date'], unit='ns', utc=True)

    def run(self, rebalancing_strategy) -> dict[str, pd.DataFrame]:
        '''
        Runs a backtest on one instrument.
        '''

        # result = {'wealth': at start of period, 'weights': at start of period, 'yield': to end of period, 'tx_cost': to end of period}
        result = {'wealth': list(),
                  'weights': list(),
                  'yield': list(),
                  'tx_cost': list()}

        for index in rebalancing_strategy.features.index[:-1]:
            prev_state = rebalancing_strategy.state
            result['wealth'].append({'time': index, 'total': prev_state.wealth})
            result['weights'].append({'time': index} | {i: value for i, value in enumerate(prev_state.weights)})

            rebalancing_strategy.update_weights()
            rebalancing_strategy.update_wealth()

            #result['yield'].append({'time': index} | {f'{col}': prev_state.weights[i] * rebalancing_strategy.features.loc[index, 'haircut_apy'] for i, col in enumerate(rebalancing_strategy.features.columns)})
            #result['tx_cost'].append({'time': index} | {f'{col}': prev_state.weights[i] * rebalancing_strategy.features.loc[index, 'haircut_apy'] for i, col in enumerate(rebalancing_strategy.features.columns)})
            #result['tx_cost'].loc[index] = rebalancing_strategy.transaction_cost(prev_state.weights, rebalancing_strategy.state.weights)

        return result


    def perf_analysis(self, df_or_path: Union[pd.DataFrame,str]) -> pd.DataFrame:
        '''
        returns quantile across folds/instument for annual_perf, sortino, drawdown
        '''
        new_values = []
        if os.path.isfile(df_or_path):
            df = pd.read_csv(df_or_path, header=list(range(6)), index_col=0)
        else:
            df = df_or_path

        for model, label, frequency in set(zip(*[df.columns.get_level_values(level) for level in ['model', 'feature', 'frequency']])):
            cumulative_pnl = df.xs((model, label, frequency, 'cumulative_pnl'),
                                   level=['model', 'feature', 'frequency', 'datatype'],
                                   axis=1)
            # remove index
            cumulative_pnl = pd.DataFrame({col: data.dropna().values
                                           for col, data in cumulative_pnl.items()})
            # annual_perf. I consider signal works if mean(annual_perf over splits and instruments) > std over the same
            annual_perf = pd.Series(cumulative_pnl.iloc[-1] / len(cumulative_pnl.index) / float(frequency) * timedelta(days=365.25).total_seconds() / ResearchEngine.data_interval.total_seconds(),
                                    name=(model, label, frequency, 'annual_perf'))


            # sortino
            pnl = cumulative_pnl.diff()
            rescaling = timedelta(days=365.25).total_seconds() / ResearchEngine.data_interval.total_seconds() / int(frequency)
            downside_dev = pnl.applymap(lambda x: min(0, x)**2).mean().apply(np.sqrt)
            sortino = pd.Series(pnl.mean() / downside_dev * np.sqrt(rescaling), name=(model, label, frequency, 'sortino'))

            # hit ratio
            hit_ratio = pnl.applymap(lambda x: x > 0).sum() / pnl.applymap(lambda x: x != 0).sum()
            hit_ratio.name = (model, label, frequency, 'hit_ratio')

            # drawdown
            drawdown = pd.Series((1 - cumulative_pnl/cumulative_pnl.cummax()).cummin().iloc[-1], name=(model, label, frequency, 'drawdown'))

            new_values.append(pd.concat([annual_perf, sortino, drawdown, hit_ratio], axis=1))

        result = pd.concat(new_values, axis=1)
        result.columns = pd.MultiIndex.from_tuples(result.columns,
                                                   names=['model', 'feature', 'frequency',
                                                          'datatype'])
        return result.describe()

