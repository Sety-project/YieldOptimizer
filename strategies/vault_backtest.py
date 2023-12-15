import logging
import os
from datetime import timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from utils.io_utils import modify_target_with_argument, dict_list_to_combinations
from research.research_engine import build_ResearchEngine, ResearchEngine
from strategies.vault_betsizing import YieldStrategy
from strategies.vault_betsizing import VaultRebalancingStrategy
from utils.sklearn_utils import entropy
from concurrent.futures import ThreadPoolExecutor, as_completed

class VaultBacktestEngine:
    def __init__(self, performance: pd.DataFrame, params: dict):
        self.start_date = pd.to_datetime(params['start_date'], unit='ns', utc=True)
        self.end_date = pd.to_datetime(params['end_date'], unit='ns', utc=True)
        self.performance = performance[(performance.index >= self.start_date) & (performance.index <= self.end_date)]

    @staticmethod
    def run(parameters: dict) -> tuple[dict, pd.DataFrame, pd.Series]:
        '''
        Runs a backtest on one instrument.
        '''
        engine = build_ResearchEngine(parameters)
        performance = pd.concat(engine.performance, axis=1)
        # backtest truncates and fillna performance to match start and end date
        backtest = VaultBacktestEngine(performance, parameters['backtest'])
        rebalancing_strategy = YieldStrategy(research_engine=engine, params=parameters['strategy'])

        result = pd.DataFrame()
        prev_index = backtest.performance.index[0]
        for index, cur_performance in backtest.performance.iterrows():
            prev_state = deepcopy(rebalancing_strategy.state)

            predicted_apys, tvl = rebalancing_strategy.predict(index)
            optimal_weights = rebalancing_strategy.optimal_weights(predicted_apys, tvl)
            step_results = rebalancing_strategy.update_wealth(optimal_weights, prev_state, prev_index, cur_performance.fillna(0.0))

            prev_index = deepcopy(index)

            new_entry = backtest.record_result(index, predicted_apys, prev_state, rebalancing_strategy, step_results)
            result = pd.concat([result, new_entry.to_frame().T], axis=0)

        return result

    @staticmethod
    def run_grid(parameter_grid: dict, parameters: dict) -> pd.DataFrame:
        # data
        perf_list: list[pd.Series] = list()
        for cur_params_override in dict_list_to_combinations(parameter_grid):
            # skip run if already in directory
            name = pd.Series(cur_params_override)
            name_to_str = ''.join(['{}_'.format(str(elem)) for elem in name]) + '_backtest.csv'
            vault_name = parameters['input_data']['dirpath'][-1].lower()
            filename = os.path.join(os.sep, os.getcwd(), "logs", vault_name, name_to_str)
            if os.path.isfile(filename):
                logging.getLogger('defillama').warning(f'{filename} already run - delete it to rerun')
                result = pd.read_csv(filename, index_col=0, header=[0, 1], parse_dates=True)
                perf = VaultBacktestEngine.perf_analysis(result)
            else:
                cur_params = modify_target_with_argument(parameters, cur_params_override)
                result = VaultBacktestEngine.run(cur_params)
                perf = VaultBacktestEngine.perf_analysis(result)
                result.to_csv(os.path.join(filename))

            perf_list.append(pd.concat([name, perf]))

        return pd.DataFrame(perf_list)

    def record_result(self, index, predicted_apys, prev_state, rebalancing_strategy, step_results) -> pd.Series:
        weights = {f'weight_{i}': weight
                   for i, weight in enumerate(prev_state.weights)}
        weights = pd.Series(weights)
        full_apy = pd.concat(rebalancing_strategy.research_engine.performance, axis=1)
        apy = rebalancing_strategy.research_engine.Y

        # yields = {f'apy_{i}': perf
        #           for i, perf in enumerate(apy.loc[index].values)}
        # pred_yields = {f'pred_apy_{i}': predicted_apy
        #                for i, predicted_apy in enumerate(predicted_apys)}
        # pred_yields = {f'apy_{i}': predicted_apy
        #                for i, predicted_apy in enumerate(predicted_apys)}
        temp = dict()
        temp['weights'] = weights.values
        temp['apy'] = apy.loc[index].values
        temp['pred_apy'] = predicted_apys
        temp['full_apy'] = full_apy.loc[index].values
        temp['dilutor'] = step_results['dilutor']

        temp = pd.DataFrame(temp).fillna(0.0)
        temp.index = self.performance.columns

        for col in ['full_apy', 'apy', 'pred_apy']:
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