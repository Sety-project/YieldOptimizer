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

        result = {'pnl': [],'weights': [],'yields': [], 'pred_yields': []}
        prev_index = self.performance.index[0]
        for index, cur_performance in self.performance.iterrows():
            prev_state = deepcopy(rebalancing_strategy.state)

            predicted_apys = rebalancing_strategy.predict(index)
            optimal_weights = rebalancing_strategy.optimal_weights(predicted_apys)
            transaction_costs, gas = rebalancing_strategy.update_wealth(optimal_weights, prev_state, prev_index, cur_performance.fillna(0.0))

            prev_index = deepcopy(index)

            self.record_result(index, predicted_apys, prev_state, rebalancing_strategy, transaction_costs, gas, result)

        return {key: pd.DataFrame(list_series) for key, list_series in result.items()}

    @staticmethod
    def run_grid(parameter_grid: dict, parameters: dict) -> pd.DataFrame:
        def modify_target_with_argument(target: dict, argument: dict) -> dict:
            result = deepcopy(target)
            if "cap" in argument:
                result["run_parameters"]["models"]["haircut_apy"]["TrivialEwmPredictor"]["params"]['cap'] = argument[
                    'cap']
            if "haflife" in argument:
                result["run_parameters"]["models"]["haircut_apy"]["TrivialEwmPredictor"]["params"]['halflife'] = \
                argument[
                    'haflife']
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
        engine: ResearchEngine = build_ResearchEngine(parameters)
        performance = pd.concat(engine.performance, axis=1)
        result: list[pd.Series] = list()
        for cur_params in dict_list_to_combinations(parameter_grid):
            new_parameter = modify_target_with_argument(parameters, cur_params)
            name = pd.Series(cur_params)

            engine = build_ResearchEngine(new_parameter)
            # backtest truncatesand fillna performance to match start and end date
            backtest = VaultBacktestEngine(performance, parameters['backtest'])

            vault_rebalancing = YieldStrategy(research_engine=engine, params=new_parameter['strategy'])
            cur_run = backtest.run(vault_rebalancing)

            # print to file
            name_to_str = ''.join(['{}_'.format(str(elem)) for elem in name]) + '_backtest'
            VaultBacktestEngine.write_results(cur_run, os.path.join(os.sep, os.getcwd(), "logs"), name_to_str)

            # insert in dict
            result.append(pd.concat([pd.Series(cur_params), backtest.perf_analysis(cur_run)]))

        return pd.DataFrame(result)

    def record_result(self, index, predicted_apys, prev_state, rebalancing_strategy, transaction_costs, gas, result):
        weights = {f'weight_{i}': weight
                   for i, weight in enumerate(prev_state.weights)}
        performance = pd.concat(rebalancing_strategy.research_engine.performance, axis=1)
        yields = {f'yield_{i}': perf
                  for i, perf in enumerate(performance.loc[index].values)}
        pred_yields = {f'pred_yield_{i}': predicted_apy
                       for i, predicted_apy in enumerate(predicted_apys)}
        temp = dict()
        temp['weights'] = pd.Series(name=index, data=weights
                                                     | {f'weight_base': prev_state.wealth - sum(weights.values())})
        eff_yield = np.dot(list(yields.values()),
                           list(weights.values())) / max(1e-8,
                                                         sum(weights.values()))
        temp['yields'] = pd.Series(name=index, data=yields | {f'eff_yield': eff_yield})
        eff_pred_yield = np.dot(list(pred_yields.values()),
                                list(weights.values())) / max(1e-8,
                                                              sum(weights.values()))
        temp['pred_yields'] = pd.Series(name=index, data=pred_yields | {f'eff_pred_yield': eff_pred_yield})
        predict_horizon = rebalancing_strategy.research_engine.label_map['haircut_apy']['horizons'][0]
        temp['pnl'] = pd.Series(name=index, data=
        {'wealth': prev_state.wealth,
         'tx_cost': transaction_costs,
         'gas': gas,
         'tracking_error': np.dot(prev_state.weights,
                                  performance.rolling(predict_horizon).mean().shift(
                                      -predict_horizon).loc[index] - predicted_apys)
                           / max(1e-8, sum(prev_state.weights))})
        for key, value in temp.items():
            result[key].append(value)

    @staticmethod
    def write_results(df: dict[str, pd.DataFrame], dirname: str, filename: str):
        '''
        writes backtest results to a csv file
        '''
        for key, value in deepcopy(df).items():
            value.columns = pd.MultiIndex.from_tuples([(key, col) for col in value.columns])
        pd.concat(df.values(), axis=1).to_csv(os.path.join(os.sep, dirname, f'{filename}.csv'))
    def perf_analysis(self, df: dict[str, pd.DataFrame]) -> pd.DataFrame:
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