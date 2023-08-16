import os
from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import scipy.optimize as opt

run_date = datetime.now()

class State:
    '''
    State of a vault. Mainly the strategies weights.
    important: base holding has no weight, but is implicit.
    so base holding = wealth - sum(weights)
    '''
    def __init__(self, weights: np.array, wealth: float):
        self.weights: np.array = weights
        self.wealth: float = wealth


class TransactionCost:
    '''
    cost to switch from a strategy to another. with optional context kwargs.
    '''
    def __init__(self, params: dict):
        self.params = params

    def __call__(self, state_0: np.array, state_1: np.array, **kwargs):
        raise NotImplementedError
    def jacobian(self, state_0: np.array, state_1: np.array, **kwargs):
        raise NotImplementedError

class HardCodedTransactionCost(TransactionCost):
    '''
    described by a pos symetric n x n matrix with 0 diagonal
    '''
    def __init__(self, params: dict):
        super().__init__(params)
        matrix = self.params['cost_matrix']
        assert matrix.shape[0] == matrix.shape[1], "need square cost matrix"
        assert matrix.transpose() == matrix, "need symetric cost matrix"
        assert all(matrix[i, i] == 0 for i in range(matrix.shape[0])), "need zero diagonal"
        assert all(matrix[i, 0] >= 0 for i in range(matrix.shape[0])), "cost is positive"

    def __call__(self, state_0: np.array, state_1: np.array, **kwargs):
        assert len(state_0) == self.params['cost_matrix'].shape[0], "cost matrix vs State mismatch"
        matrix = self.params['cost_matrix']
        raise NotImplementedError


class TransactionCostThroughBase(TransactionCost):
    '''
    we trade through base (WETH) so only a n-1 vector is supplied as cost_vector
    '''

    def __init__(self, params: dict):
        self.vector = params['cost_vector']
        super().__init__(params)

    def __call__(self, state_0: np.array, state_1: np.array, **kwargs):
        assert len(state_0) == len(self.params['cost_vector']), "cost matrix vs State mismatch"
        return np.dot(self.vector, np.abs((state_1 - state_0)))

    def jacobian(self, state_0: np.array, state_1: np.array, **kwargs):
        assert len(state_0) == len(self.params['cost_vector']), "cost matrix vs State mismatch"
        return np.array([cost * (1 if state_1[i] > state_0[i] else -1) for i, cost in enumerate(self.vector)])


class VaultRebalancingStrategy(ABC):
    '''
    TODO: inherit from forked backtesting.py
    '''
    def __init__(self, params: dict):
        self.parameters = params
    @abstractmethod
    def update_weights(self):
        raise NotImplementedError


class YieldStrategy(VaultRebalancingStrategy):
    '''
    convex optimization of yield
    '''
    def __init__(self, params: dict, features: pd.DataFrame, fitted_model):
        '''
        weights excludes base asset balances. Initialize at 0 (all base)
        wealth: base holding = wealth - sum(weights). Initialize at 1.
        '''
        super().__init__(params)
        self.parameters = params
        self.features: pd.DataFrame = features
        self.fitted_model = fitted_model

        self.index: datetime = self.features.index[0]
        self.state: State = State(weights=np.zeros(features.shape[1]), wealth=1.0)

        if 'cost' in params:
            self.transaction_cost = TransactionCostThroughBase({
                'cost_vector': params['cost']*np.ones(features.shape[1])/params['assumed_holding_yrs']})
        elif 'cost_matrix' in params:
            raise NotImplementedError
            self.transaction_cost = TransactionCost(params)
    def update_weights(self) -> None:
        '''
        fit_predict on current features -> predicted_apy -> convex optimization under wealth constraint + cost/risk penalty
        '''
        self.fitted_model.fit(self.features[self.features.index<=self.index])
        predicted_apys = self.fitted_model.distribution.mean

        ### objective is APY - txcost (- risk?)
        objective = lambda x: -(
                np.dot(x, predicted_apys)
                - self.transaction_cost(self.state.weights, x)
        ) #TODO could easily add vol penalty
        objective_jac = lambda x: -(
                predicted_apys
                - self.transaction_cost.jacobian(self.state.weights, x)
        )
        constraints = [{'type': 'ineq',
                        'fun': lambda x: self.state.wealth - np.sum(x),
                        'jac': lambda x: -np.ones(len(x))}]

        bounds = opt.Bounds(lb=np.zeros(len(self.state.weights)),
                            ub=self.state.wealth * np.ones(len(self.state.weights)))

        # --------- verbose callback function: breaks down pnl during optimization
        def callbackF(x, print_with_flag=None):
            if print_with_flag is not None:
                progress_display = pd.DataFrame(pd.concat([
                    pd.Series({
                        'predicted_apys': np.dot(x, predicted_apys),
                        'tx_cost': self.transaction_cost(self.state.weights, x),
                        'wealth_constraint': constraints[0]['fun'](x),
                        'success': print_with_flag
                    }),
                    pd.Series(x)
                ]))  # used .append
                pfoptimizer_path = os.path.join(os.sep, os.getcwd(), "logs")
                if not os.path.exists(pfoptimizer_path):
                    os.umask(0)
                    os.makedirs(pfoptimizer_path, mode=0o777)
                global run_date
                pfoptimizer_filename = os.path.join(pfoptimizer_path, "{}_paths.csv".format(run_date.strftime("%Y%m%d-%H%M%S")))
                progress_display.T.to_csv(pfoptimizer_filename,
                                                           mode='a',
                                                           header=not os.path.exists(pfoptimizer_filename))
            return []

        if 'warm_start' in self.parameters:
            x1 = self.state.weights
        else:
            x1 = np.zeros(len(self.state.weights))

        if 'verbose' in self.parameters:
            callbackF(x1, 'initial')

        finite_diff_rel_step = self.parameters['finite_diff_rel_step'] if 'finite_diff_rel_step' in self.parameters else 1e-3
        res = opt.minimize(objective, x1, method='SLSQP', jac=objective_jac,
                           constraints=constraints,  # ,loss_tolerance_constraint
                           bounds=bounds,
                           callback=(lambda x: callbackF(x,
                                                         'interim' if 'verbose' in self.parameters else None)),
                           options={'ftol': 1e-2, 'disp': False, 'finite_diff_rel_step': finite_diff_rel_step,
                                    'maxiter': 50 * len(x1)})
        if not res['success']:
            # cheeky ignore that exception:
            # https://github.com/scipy/scipy/issues/3056 -> SLSQP is unfomfortable with numerical jacobian when solution is on bounds, but in fact does converge.
            violation = - min([constraint['fun'](res['x']) for constraint in constraints])
            if res['message'] == 'Iteration limit reached':
                logging.getLogger('pfoptimizer').warning(res['message'] + '...but SLSQP is uncomfortable with numerical jacobian when solution is on bounds, but in fact does converge.')
            elif res['message'] == "Inequality constraints incompatible" and violation < self.state.wealth / 100:
                logging.getLogger('pfoptimizer').warning(res['message'] + '...but only by' + str(violation))
            else:
                logging.getLogger('pfoptimizer').critical(res['message'])

        if 'verbose' in self.parameters:
            callbackF(res['x'], res['message'])

        self.state.weights = res['x']
        self.index = next((t for t in self.features.index if t > self.index), None)

        if self.index is None:
            raise StopIteration

    def update_wealth(self):
        '''
        update wealth from yields.
        '''
        prev_index = self.features[self.features.index<self.index].index[-1]
        dt = (self.index - prev_index).total_seconds() / timedelta(days=365).total_seconds()
        yields_dt = self.features.loc[self.index].values * dt
        base_yield_dt = 0.0
        x = self.state
        x.wealth *= np.exp(np.dot(x.weights, yields_dt) + (x.wealth - sum(x.weights)) * base_yield_dt)
        return
