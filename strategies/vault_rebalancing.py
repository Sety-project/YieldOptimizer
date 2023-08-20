import os
from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import sklearn
import cvxpy as cp
import io
from contextlib import redirect_stdout
from research.research_engine import ResearchEngine

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
    def __call__(self, state_0: np.array, state_1: np.array, **kwargs):
        assert len(state_0) == len(self.params['cost_vector']), "cost matrix vs State mismatch"
        return np.dot(self.params['cost_vector'], np.abs(state_1 - state_0))

    def jacobian(self, state_0: np.array, state_1: np.array, **kwargs):
        assert len(state_0) == len(self.params['cost_vector']), "cost matrix vs State mismatch"
        return np.array([cost * (1 if state_1[i] > state_0[i] else -1) for i, cost in enumerate(self.params['cost_vector'])])


class VaultRebalancingStrategy(ABC):
    '''
    TODO: inherit from forked backtesting.py
    '''
    def __init__(self, params: dict, features: pd.DataFrame, research_engine: ResearchEngine):
        self.parameters = params
        self.features: pd.DataFrame = features
        self.research_engine = research_engine

        self.index: datetime = self.features.index[0]
        self.state: State = State(weights=np.zeros(self.features.shape[1]), wealth=self.parameters['initial_wealth'])

        if 'cost' in params:
            assumed_holding_days = research_engine.label_map['haircut_apy']['horizons']
            self.transaction_cost = TransactionCostThroughBase({
                'cost_vector': np.full(features.shape[1], params['cost']/(assumed_holding_days[0]/365.0))})
        elif 'cost_matrix' in params:
            raise NotImplementedError
            self.transaction_cost = TransactionCost(params)
    @abstractmethod
    def predict(self, model: sklearn.base.BaseEstimator):
        raise NotImplementedError
    @abstractmethod
    def update_weights(self):
        raise NotImplementedError
    @abstractmethod
    def update_wealth(self):
        raise NotImplementedError
class YieldStrategy(VaultRebalancingStrategy):
    '''
    convex optimization of yield
    '''
    def __init__(self, params: dict, features: pd.DataFrame, research_engine: ResearchEngine):
        '''
        weights excludes base asset balances. Initialize at 0 (all base)
        wealth: base holding = wealth - sum(weights). Initialize at 1.
        '''
        super().__init__(params, features, research_engine)
        self.progress_display = pd.DataFrame()
        self.current_time = datetime.now()

    def update_weights(self, predicted_apys: np.array) -> None:
        '''
        fit_predict on current features -> predicted_apy -> convex optimization under wealth constraint + cost/risk penalty
        '''
        ####### CVXPY version #######
        res = self.solve_cvx_problem(predicted_apys,
        solver=getattr(cp, 'ECOS'), verbose=False, warm_start=False,
                                     max_iters=1000,
                                     abstol=1e-3,
                                     reltol=1e-3,
                                     feastol=1e-3)
        #res2 = self.solve_scipy_problem(predicted_apys)

        self.state.weights = res['x']
        self.index = next((t for t in self.features.index if t > self.index), None)
        return

    def predict(self, model: sklearn.base.BaseEstimator):
        history = self.features[self.features.index <= self.index]
        predicted_apys = model.predict_proba(history).mean
        return predicted_apys

    # def solve_scipy_problem(self, predicted_apys: np.array):
    #     ### objective is APY - txcost (- risk?)
    #     # TODO could easily add vol penalty
    #     # TODO could easily discount apy by our mktimpact (linear dilution of apy)
    #     objective = lambda x: -(
    #         np.dot(x, predicted_apys)
    #         #                - self.transaction_cost(self.state.weights, x)
    #     )
    #     objective_jac = lambda x: -(
    #         predicted_apys
    #         #               - self.transaction_cost.jacobian(self.state.weights, x)
    #     )
    #     constraints = [{'type': 'ineq',
    #                     'fun': lambda x: self.state.wealth * (1.0 - self.parameters['base_buffer']) - np.sum(x),
    #                     'jac': lambda x: -np.ones(self.features.shape[1])}]
    #     bounds = opt.Bounds(lb=np.zeros(self.features.shape[1]),
    #                         ub=np.full(self.features.shape[1], self.state.wealth))
    #
    #     # --------- verbose callback function: breaks down pnl during optimization
    #     def callbackF(x, details=None, print_with_flag=None):
    #         new_progress = pd.Series({
    #                                      'predicted_apy': np.dot(x, predicted_apys) / max(1e-8, np.sum(x)),
    #                                      'tx_cost': self.transaction_cost(self.state.weights, x),
    #                                      'wealth_constraint (=base weight)': constraints[0]['fun'](x),
    #                                      'status': print_with_flag,
    #                                      'jac_error': opt.check_grad(objective, objective_jac, x),
    #                                  }
    #                                  | {f'weight_{i}': value for i, value in enumerate(x)}
    #                                  | {f'pred_apy_{i}': value for i, value in enumerate(predicted_apys)})
    #                             #TODO:| {f'spot_apy_{i}': value for i, value in enumerate(history.iloc[-1].values)})
    #         pfoptimizer_path = os.path.join(os.sep, os.getcwd(), "logs")
    #         if not os.path.exists(pfoptimizer_path):
    #             os.umask(0)
    #             os.makedirs(pfoptimizer_path, mode=0o777)
    #         pfoptimizer_filename = os.path.join(os.sep, pfoptimizer_path, "{}_optimization.csv".format(
    #             self.current_time.strftime("%Y%m%d-%H%M%S")))
    #         self.progress_display = pd.concat([self.progress_display, new_progress], axis=1)
    #         self.progress_display.T.to_csv(pfoptimizer_filename, mode='w')
    #
    #     if 'warm_start' in self.parameters and self.parameters['warm_start']:
    #         x1 = self.state.weights
    #     else:
    #         x1 = np.zeros(self.features.shape[1])
    #     if 'verbose' in self.parameters and self.parameters['verbose']:
    #         callbackF(x1, details=None, print_with_flag='initial')
    #     ftol = self.parameters['ftol'] if 'ftol' in self.parameters else 1e-4
    #     finite_diff_rel_step = self.parameters[
    #         'finite_diff_rel_step'] if 'finite_diff_rel_step' in self.parameters else 1e-2
    #     method = 'SLSQP'
    #     if method == 'SLSQP':
    #         res = opt.minimize(objective, x1, method=method, jac=objective_jac,
    #                            constraints=constraints,  # ,loss_tolerance_constraint
    #                            bounds=bounds,
    #                            callback=(lambda x: callbackF(x, details=None, print_with_flag='interim')) if (
    #                                        'verbose' in self.parameters and self.parameters['verbose']) else None,
    #                            options={'ftol': ftol, 'disp': False, 'finite_diff_rel_step': finite_diff_rel_step,
    #                                     'maxiter': 50 * self.features.shape[1]})
    #     elif method == 'trust-constr':
    #         res = opt.minimize(objective, x1, method=method, jac=objective_jac,
    #                            constraints=[LinearConstraint(np.array([np.ones(self.features.shape[1])]),
    #                                                          [0],
    #                                                          [self.state.wealth * (
    #                                                                      1 - self.parameters['base_buffer'])])],
    #                            # ,loss_tolerance_constraint
    #                            bounds=bounds,
    #                            callback=(
    #                                lambda x, details: callbackF(x, details=details, print_with_flag='interim')) if (
    #                                    'verbose' in self.parameters and self.parameters['verbose']) else None,
    #                            options={'gtol': ftol, 'disp': False,
    #                                     'maxiter': 50 * self.features.shape[1]})
    #     if not res['success']:
    #         # cheeky ignore that exception:
    #         # https://github.com/scipy/scipy/issues/3056 -> SLSQP is unfomfortable with numerical jacobian when solution is on bounds, but in fact does converge.
    #         violation = - min([constraint['fun'](res['x']) for constraint in constraints])
    #         if res['message'] == 'Iteration limit reached':
    #             logging.getLogger('pfoptimizer').warning(res[
    #                                                          'message'] + '...but SLSQP is uncomfortable with numerical jacobian when solution is on bounds, but in fact does converge.')
    #         elif res['message'] == "Inequality constraints incompatible" and violation < self.state.wealth / 100:
    #             logging.getLogger('pfoptimizer').warning(res['message'] + '...but only by' + str(violation))
    #         else:
    #             logging.getLogger('pfoptimizer').critical(res['message'])
    #     if 'verbose' in self.parameters and self.parameters['verbose']:
    #         callbackF(res['x'], details=None, print_with_flag=res['message'])
    #     return res

    def solve_cvx_problem(self, predicted_apys, **kwargs):
        # Define the variables and parameters
        x = cp.Variable(shape=self.features.shape[1], nonneg=True, value=self.state.weights)
        a = cp.Parameter(shape=self.features.shape[1], nonneg=True, value=predicted_apys)
        x0 = cp.Parameter(shape=self.features.shape[1], nonneg=True, value=self.state.weights)
        cost = cp.Parameter(shape=self.features.shape[1], nonneg=True, value=self.transaction_cost.params['cost_vector'])
        max_x = cp.Parameter(value=self.state.wealth * (1.0 - self.parameters['base_buffer']))
        # Define the DCP expression. Trick y to make DPP compliant with y ?
        objective = cp.Minimize(-a @ x + cost @ cp.abs(x - x0))
        constraints = [cp.sum(x) <= max_x]

        problem = cp.Problem(objective, constraints)
        # Solve the problem and print stdout
        with io.StringIO() as buf, redirect_stdout(buf):
            problem.solve(**kwargs)
            solver_comments = buf.getvalue()

        return {'success': problem.status,
                'message': solver_comments,
                'y': problem.value if problem.status == 'optimal' else None,
                'x': x.value if problem.status == 'optimal' else None}

    def update_wealth(self):
        '''
        update wealth from yields.
        '''
        prev_index = self.features[self.features.index<self.index].index[-1]

        dt = (self.index - prev_index) / timedelta(days=365)
        yields_dt = self.features.loc[self.index].values * dt
        base_yield_dt = 0.0

        x = self.state
        base_weight = x.wealth - np.sum(x.weights)

        x.weights *= np.exp(yields_dt)
        base_weight *= np.exp(base_yield_dt)
        x.wealth = np.sum(x.weights) + base_weight

        return
