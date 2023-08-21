import os
from datetime import timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from research.research_engine import ResearchEngine
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

            prev_index = index

            self.record_result(index, predicted_apys, prev_state, rebalancing_strategy, transaction_costs, gas, result)

        return {key: pd.DataFrame(list_series) for key, list_series in result.items()}

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