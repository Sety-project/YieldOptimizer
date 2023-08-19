import os
from datetime import timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from strategies.vault_rebalancing import VaultRebalancingStrategy
from strategies.vault_rebalancing import run_date
from utils.sklearn_utils import entropy

class VaultBacktestEngine:
    def __init__(self, params: dict):
        self.parameters = params
        self.parameters['start_date'] = pd.to_datetime(self.parameters['start_date'], unit='ns', utc=True)
        self.parameters['end_date'] = pd.to_datetime(self.parameters['end_date'], unit='ns', utc=True)

    def run(self, rebalancing_strategy: VaultRebalancingStrategy) -> pd.DataFrame:
        '''
        Runs a backtest on one instrument.
        '''

        result = {'pnl': [],'weights': [],'yields': []}
        indices = rebalancing_strategy.features[
            (rebalancing_strategy.features.index>=self.parameters['start_date'])
        & (rebalancing_strategy.features.index <= self.parameters['end_date'])].index
        for index in indices:
            prev_state = deepcopy(rebalancing_strategy.state)
            rebalancing_strategy.update_weights()
            rebalancing_strategy.update_wealth()

            weights = {f'weight_{i}': prev_state.weights[i]
                       for i, _ in enumerate(rebalancing_strategy.features.columns)}
            yields = {f'yield_{i}': rebalancing_strategy.features.loc[index].iloc[i]
                      for i, _ in enumerate(rebalancing_strategy.features.columns)}

            temp = dict()
            temp['pnl'] = pd.Series(name=index, data=
            {'wealth': prev_state.wealth,
             'tx_cost': rebalancing_strategy.transaction_cost(prev_state.weights,
                                                              rebalancing_strategy.state.weights)})
            temp['weights'] = pd.Series(name=index, data=weights
                                                         |{f'weight_base': prev_state.wealth - sum(weights.values())})
            temp['yields'] = pd.Series(name=index, data=yields
                                                        |{f'eff_yield': np.dot(list(yields.values()),
                                                                                list(weights.values())) / max(1e-8,
                                                                                                              sum(weights.values()))})
            for key, value in temp.items():
                result[key].append(value)

        return {key: pd.DataFrame(list_series) for key, list_series in result.items()}
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