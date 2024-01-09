import os
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd

from research.research_engine import ResearchEngine
from strategies.cta_betsizing import SingleAssetStrategy


class BacktestEngine:
    def __init__(self, params: dict, delta_strategy: SingleAssetStrategy):
        self.parameters = params
        self.delta_strategy = delta_strategy

    def backtest(self, features, fitted_model, frequency: int, rebalance_times, history) -> dict[str, pd.Series]:
        '''
        Runs a backtest on one instrument.
        '''
        # features -> prediction -> delta
        delta = self.delta_strategy.run(features, fitted_model)
        # apply log increment to delta
        tx_cost = self.parameters['tx_cost'] * delta.diff().apply(np.abs)
        performance = (history.shift(-frequency - ResearchEngine.execution_lag) / history.shift(-ResearchEngine.execution_lag) - 1)[rebalance_times]
        cumulative_pnl = (delta * performance - tx_cost).cumsum().shift(1)

        return {'cumulative_performance': history[delta.index]/history[delta.index[0]]-1,
                'delta': delta,
                'tx_cost': tx_cost,
                'cumulative_pnl': cumulative_pnl}

    def backtest_all(self, prediction_engine: ResearchEngine) -> pd.DataFrame():
        '''
        Runs a backtest per instrument.
        '''
        backtests = []
        for (raw_feature, frequency), label_df in prediction_engine.Y.groupby(level=['feature', 'window'], axis=1):
            for model_name, model_params in prediction_engine.run_parameters['models'][raw_feature].items():
                for _, _, _, split_index in filter(lambda x: x[0] == raw_feature
                                                    and x[1] == frequency
                                                    and x[2] == model_name,
                                          prediction_engine.fitted_model.keys()):
                    fitted_model, test_index = prediction_engine.fitted_model[(raw_feature, frequency, model_name, split_index)]
                    for instrument in prediction_engine.input_data['selected_instruments']:
                        rebalance_times = prediction_engine.X.iloc[test_index].index[::frequency]
                        features = prediction_engine.X.xs(instrument, level='instrument', axis=1).loc[rebalance_times]
                        temp = self.backtest(features, fitted_model,
                                             frequency, rebalance_times,
                                             prediction_engine.performance[instrument])
                        temp = {(model_name, split_index, instrument, raw_feature, frequency, key): value for key,value in temp.items()}

                        # add to backtest, putting all folds on a relative time axis
                        new_backtest = pd.DataFrame(temp)
                        new_backtest.index = new_backtest.index - rebalance_times[0]
                        backtests.append(new_backtest)

        backtest = pd.concat(backtests, axis=1, join='outer')
        backtest.columns = pd.MultiIndex.from_tuples(backtest.columns, names=['model', 'split_index', 'instrument', 'feature', 'frequency', 'datatype'])

        return backtest

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

