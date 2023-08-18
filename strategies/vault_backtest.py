import os
from datetime import timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from strategies.vault_rebalancing import VaultRebalancingStrategy
from strategies.vault_rebalancing import run_date

class VaultBacktestEngine:
    def __init__(self, params: dict):
        self.parameters = params
        self.parameters['start_date'] = pd.to_datetime(self.parameters['start_date'], unit='ns', utc=True)
        self.parameters['end_date'] = pd.to_datetime(self.parameters['end_date'], unit='ns', utc=True)

    def run(self, rebalancing_strategy: VaultRebalancingStrategy) -> pd.DataFrame:
        '''
        Runs a backtest on one instrument.
        '''

        result = pd.DataFrame()
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

            temp = pd.Series(name=index,data=
                              {'wealth': prev_state.wealth,
                              'tx_cost': rebalancing_strategy.transaction_cost(prev_state.weights,
                                                                               rebalancing_strategy.state.weights)}
                             | weights
                             | {f'weight_total': sum(weights.values())}
                             | yields
                             | {f'yield': np.dot(list(yields.values()), list(weights.values()))/max(1e-8,sum(weights.values()))})
            result = pd.concat([result, temp], axis=1)

        pfoptimizer_path = os.path.join(os.sep, os.getcwd(), "logs")
        if not os.path.exists(pfoptimizer_path):
            os.umask(0)
            os.makedirs(pfoptimizer_path, mode=0o777)

        pfoptimizer_filename = os.path.join(pfoptimizer_path, "{}_result.csv".format(run_date.strftime("%Y%m%d-%H%M%S")))
        result.T.to_csv(pfoptimizer_filename,
                                  mode='w')
        return result.T

    def perf_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        returns performance metrics for a backtest: perf, tx_cost, avg gini
        '''
        dt = ((df.index.max() - df.index.min())/timedelta(days=365))

        weights = df[[col for col in df.columns if 'weight_' in col and col != 'weight_total']]
        weights.loc[:,'weight_base'] = df['wealth'] - weights.sum(axis=1)
        # sum p log p / max_entropy (=)
        entropy = weights.div(df['wealth'], axis=0).apply(lambda x: -np.log(x)*x).sum(axis=1)/np.log(weights.shape[1])

        result = pd.Series({'perf': np.log(df['wealth'].iloc[-1]/df['wealth'].iloc[0])/dt,
                   'tx_cost': df['tx_cost'].sum()/df['wealth'].iloc[0]/dt,
                   'avg_entropy': entropy.mean()})

        return result