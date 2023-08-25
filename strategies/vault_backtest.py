import os
from datetime import timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from research.research_engine import build_ResearchEngine, ResearchEngine
from strategies.vault_rebalancing import YieldStrategy
from strategies.vault_rebalancing import VaultRebalancingStrategy
from utils.sklearn_utils import entropy

class VaultBacktestEngine:
    def __init__(self, performance: pd.DataFrame, params: dict):
        self.start_date = pd.to_datetime(params['start_date'], unit='ns', utc=True)
        self.end_date = pd.to_datetime(params['end_date'], unit='ns', utc=True)
        self.performance = performance[(performance.index >= self.start_date) & (performance.index <= self.end_date)]

    def run(self, rebalancing_strategy: VaultRebalancingStrategy) -> dict[str, pd.DataFrame]:
        '''
        Runs a backtest on one instrument.
        '''

        result = pd.DataFrame()
        prev_index = self.performance.index[0]
        for index, cur_performance in self.performance.iterrows():
            prev_state = deepcopy(rebalancing_strategy.state)

            predicted_apys = rebalancing_strategy.predict(index)
            optimal_weights = rebalancing_strategy.optimal_weights(predicted_apys)
            transaction_costs, gas = rebalancing_strategy.update_wealth(optimal_weights, prev_state, prev_index, cur_performance.fillna(0.0))

            prev_index = deepcopy(index)

            new_entry = self.record_result(index, predicted_apys, prev_state, rebalancing_strategy, transaction_costs, gas)
            result = result.append(new_entry.to_frame().T)

        return result

    @staticmethod
    def run_grid(parameter_grid: dict, parameters: dict) -> pd.DataFrame:
        def modify_target_with_argument(target: dict, argument: dict) -> dict:
            result = deepcopy(target)
            if "cap" in argument:
                result["run_parameters"]["models"]["haircut_apy"]["TrivialEwmPredictor"]["params"]['cap'] = argument[
                    'cap']
            if "haflife" in argument:
                result["run_parameters"]["models"]["haircut_apy"]["TrivialEwmPredictor"]["params"]['halflife'] = argument['haflife']
            if "cost" in argument:
                result['strategy']['cost'] = argument['cost']
            if "gas" in argument:
                result['strategy']['gas'] = argument['gas']
            if "base_buffer" in argument:
                result['strategy']['base_buffer'] = argument['base_buffer']
            if "concentration_limit" in argument:
                result['strategy']['concentration_limit'] = argument['concentration_limit']
            if "assumed_holding_days" in argument:
                result["label_map"]["haircut_apy"]["horizons"] = [argument['assumed_holding_days']]
            return result

        def dict_list_to_combinations(d: dict) -> list[pd.DataFrame]:
            keys = d.keys()
            values = d.values()
            combinations = [dict(zip(keys, combination)) for combination in product(*values)]
            return combinations

        # data
        result: list[pd.Series] = list()
        for cur_params in dict_list_to_combinations(parameter_grid):
            new_parameter = modify_target_with_argument(parameters, cur_params)
            name = pd.Series(cur_params)

            engine = build_ResearchEngine(new_parameter)
            performance = pd.concat(engine.performance, axis=1)
            # backtest truncatesand fillna performance to match start and end date
            backtest = VaultBacktestEngine(performance, parameters['backtest'])

            vault_rebalancing = YieldStrategy(research_engine=engine, params=new_parameter['strategy'])
            cur_run = backtest.run(vault_rebalancing)

            # print to file
            name_to_str = ''.join(['{}_'.format(str(elem)) for elem in name]) + '_backtest.csv'
            cur_run.to_csv(os.path.join(os.sep, os.getcwd(), "logs", name_to_str))

            # insert in dict
            result.append(pd.concat([pd.Series(cur_params), backtest.perf_analysis(cur_run)]))

        return pd.DataFrame(result)

    def record_result(self, index, predicted_apys, prev_state, rebalancing_strategy, transaction_costs, gas) -> pd.Series:
        weights = {f'weight_{i}': weight
                   for i, weight in enumerate(prev_state.weights)}
        weights = pd.Series(weights)
        full_apy = pd.concat(rebalancing_strategy.research_engine.performance, axis=1)
        haircut_apy = rebalancing_strategy.research_engine.Y

        # yields = {f'haircut_apy_{i}': perf
        #           for i, perf in enumerate(haircut_apy.loc[index].values)}
        # pred_yields = {f'pred_haircut_apy_{i}': predicted_apy
        #                for i, predicted_apy in enumerate(predicted_apys)}
        # pred_yields = {f'apy_{i}': predicted_apy
        #                for i, predicted_apy in enumerate(predicted_apys)}
        temp = dict()
        temp['weights'] = weights.values
        temp['haircut_apy'] = haircut_apy.loc[index].values
        temp['pred_haircut_apy'] = predicted_apys
        temp['full_apy'] = full_apy.loc[index].values

        temp = pd.DataFrame(temp).fillna(0.0)

        for col in ['full_apy', 'haircut_apy', 'pred_haircut_apy']:
            temp.loc['total', col] = (temp[col] * temp['weights']).sum() / temp['weights'].apply(lambda x: np.clip(x, a_min= 1e-8, a_max=None)).sum()
        temp.loc['total', 'weights'] = prev_state.wealth - weights.sum()

        predict_horizon = rebalancing_strategy.research_engine.label_map['haircut_apy']['horizons'][0]
        pnl = pd.DataFrame({'pnl':
        {'wealth': prev_state.wealth,
         'tx_cost': transaction_costs,
         'gas': gas,
         'tracking_error': np.dot(prev_state.weights,
                                  haircut_apy.rolling(predict_horizon).mean().shift(
                                      -predict_horizon).loc[index] - predicted_apys)
                           / max(1e-8, sum(prev_state.weights))}})
        new_entry: pd.Series = pd.concat([temp.unstack(), pnl.unstack()], axis=0)
        new_entry.name = index
        return new_entry

    def perf_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
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