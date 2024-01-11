import os
from datetime import timedelta
import os
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
from streamlit.delta_generator import DeltaGenerator

from research.research_engine import build_ResearchEngine, FileData
from strategies.vault_betsizing import YieldStrategy
from utils.io_utils import modify_target_with_argument
from utils.sklearn_utils import entropy


class VaultBacktestEngine:
    weight_tolerance = 0.001
    def __init__(self, performance: pd.DataFrame, params: dict):
        self.start_date = pd.to_datetime(params['start_date'], unit='ns', utc=True)
        self.end_date = pd.to_datetime(params['end_date'], unit='ns', utc=True)
        self.performance = performance[(performance.index >= self.start_date) & (performance.index <= self.end_date)]

    @staticmethod
    def run(parameters: dict, data: FileData, progress_bar: DeltaGenerator = None) -> pd.DataFrame:
        '''
        Runs a backtest on one instrument.
        :param data:
        '''
        engine = build_ResearchEngine(parameters, data)
        performance = pd.concat(engine.performance, axis=1)
        # backtest truncates and fillna performance to match start and end date
        backtest = VaultBacktestEngine(performance, parameters['backtest'])
        rebalancing_strategy = YieldStrategy(research_engine=engine, params=parameters['strategy'])

        result = pd.DataFrame()
        prev_index = backtest.performance.index[0]
        for i, (index, cur_performance) in enumerate(backtest.performance.iterrows()):
            prev_state = deepcopy(rebalancing_strategy.state)

            predicted_apys, tvl = rebalancing_strategy.predict(index)
            optimal_weights = rebalancing_strategy.optimal_weights(predicted_apys, tvl)
            step_results = rebalancing_strategy.update_wealth(optimal_weights, prev_state, prev_index, cur_performance.fillna(0.0))

            prev_index = deepcopy(index)

            new_entry = backtest.record_result(index, predicted_apys, prev_state, rebalancing_strategy, step_results)
            result = pd.concat([result, new_entry.to_frame().T], axis=0)
            if progress_bar:
                progress_bar.progress(value=(i+1) / len(backtest.performance), text=f'Backtesting {index}')

        pool_max = result.xs(key='weights', level=0, axis=1).max()
        negligible_pools = pool_max[pool_max < parameters['strategy']['initial_wealth']*VaultBacktestEngine.weight_tolerance].index
        for col in result.columns:
            if col[1] in negligible_pools:
                result.drop(col, axis=1, inplace=True)

        return result

    @staticmethod
    def run_grid(parameter_grid: list[dict], parameters: dict, data: FileData, progress_bar1: DeltaGenerator = None, progress_bar2: DeltaGenerator = None) -> dict:
        '''
        returns:
            - runs = dict of backtest results for each parameter_grid
            - grid = performance metrics for all runs: perf, tx_cost, avg gini
        '''
        run_name = 'latest'
        dirname = os.path.join(os.sep, os.getcwd(), "logs", run_name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        result: dict[str, pd.DataFrame] = {}
        perf_list: list[pd.DataFrame] = []
        for i, cur_params_override in enumerate(parameter_grid):
            name_to_str = '_'.join([f'{str(elem)}' for elem in cur_params_override.values()])

            filename = os.path.join(os.sep, dirname, f'{name_to_str}_backtest.csv')

            cur_params = modify_target_with_argument(parameters, cur_params_override)
            result[name_to_str] = VaultBacktestEngine.run(cur_params, data, progress_bar2)
            perf = VaultBacktestEngine.perf_analysis(result[name_to_str])
            result[name_to_str].to_csv(os.path.join(filename))

            perf_list.append(pd.concat([pd.Series(cur_params_override), perf]))
            if progress_bar1:
                progress_bar1.progress(value=(i+1) / len(parameter_grid), text=f'Run {i+1} out of {len(parameter_grid)}')

        return {'grid': pd.DataFrame(perf_list), 'runs': result}

    def record_result(self, index, predicted_apys, prev_state, rebalancing_strategy, step_results) -> pd.Series:
        weights = {f'weight_{i}': weight
                   for i, weight in enumerate(prev_state.weights)}
        weights = pd.Series(weights)
        apy = rebalancing_strategy.research_engine.X.xs(key=('apy', 'as_is'), axis=1, level=('feature', 'window'))
        apyReward = rebalancing_strategy.research_engine.X.xs(key=('apyReward', 'as_is'), axis=1, level=('feature', 'window'))

        # yields = {f'apy_{i}': perf
        #           for i, perf in enumerate(apy.loc[index].values)}
        # pred_yields = {f'pred_apy_{i}': predicted_apy
        #                for i, predicted_apy in enumerate(predicted_apys)}
        # pred_yields = {f'apy_{i}': predicted_apy
        #                for i, predicted_apy in enumerate(predicted_apys)}
        temp = dict()
        temp['weights'] = weights.values
        temp['pred_apy'] = predicted_apys
        temp['apy'] = apy.loc[index].values
        temp['apyReward'] = apyReward.loc[index].values
        temp['dilutor'] = step_results['dilutor']

        temp = pd.DataFrame(temp).fillna(0.0)
        temp.index = self.performance.columns

        for col in ['apy', 'apyReward', 'pred_apy']:
            temp.loc['total', col] = (temp[col] * temp['weights']).sum() / temp['weights'].apply(lambda x: np.clip(x, a_min= 1e-8, a_max=None)).sum()
        temp.loc['total', 'weights'] = prev_state.wealth - weights.sum()

        predict_horizon = rebalancing_strategy.research_engine.label_map['apy']['horizons'][0]
        pnl = pd.DataFrame({'pnl':
                                {'wealth': prev_state.wealth,
                                 'gas': step_results['gas'],
                                 'tx_cost': step_results['transaction_costs'],
                                 'tracking_error': np.dot(prev_state.weights,
                                                          apy.rolling(predict_horizon).mean().shift(
                                                              -predict_horizon).loc[index] - predicted_apys)
                                                   / max(1e-8, sum(prev_state.weights))}})
        new_entry: pd.Series = pd.concat([temp.unstack(), pnl.unstack()], axis=0)
        new_entry.name = index
        return new_entry

    @staticmethod
    def perf_analysis(df: pd.DataFrame) -> pd.DataFrame:
        '''
        returns performance metrics for a backtest: perf, tx_cost, avg gini
        '''

        weights = df['weights']
        dt = ((weights.index.max() - weights.index.min())/timedelta(days=365))

        # sum p log p / max_entropy (=)
        entr = entropy(weights)

        pnl = df['pnl']
        result = pd.Series({'perf': np.log(pnl['wealth'].iloc[-1]/pnl['wealth'].iloc[0])/dt,
                   'tx_cost': pnl['tx_cost'].sum()/pnl['wealth'].iloc[0]/dt,
                   'avg_entropy': entr.mean()})

        return result