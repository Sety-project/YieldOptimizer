import copy
import io
import logging
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta, datetime
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import sklearn
from cvxpy import error as cvxerror

from research.research_engine import ResearchEngine
from utils.api_utils import build_logging

run_date = datetime.now()
build_logging('defillama')

class State:
    '''
    State of a vault. Mainly the strategies weights.
    important: base holding has no weight, but is implicit.
    so base holding = wealth - sum(weights)
    '''
    def __init__(self, weights: np.array, wealth: float):
        self.weights: np.array = weights
        self.wealth: float = wealth


class VaultRebalancingStrategy(ABC):
    '''
    stateful (self.index) object that implements a rebalancing strategy.
    has a research engine to get features and fitted models and performance.
    '''
    def __init__(self, research_engine: ResearchEngine, params: dict):
        self.parameters = params
        self.research_engine = research_engine

        N = len(research_engine.performance)
        self.state: State = State(weights=np.zeros(N),
                                  wealth=self.parameters['initial_wealth'])

        if 'cost' in params:
            self.cost_wealth = np.full(N, params['cost'])
            assumed_holding_days = research_engine.label_map['apy']['horizons']
            if len(assumed_holding_days) > 1:
                raise NotImplementedError
            if assumed_holding_days[0] is not None:
                self.cost_optimization = np.full(N, params['cost']/(assumed_holding_days[0]/365.0))
            else:
                self.cost_optimization = np.zeros(N)
        else:
            raise NotImplementedError

    @abstractmethod
    def predict(self, index: datetime) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def optimal_weights(self):
        raise NotImplementedError

    @abstractmethod
    def update_wealth(self, new_weights: np.array, prev_state: State, prev_index: Optional[datetime],
                      cur_performance: pd.Series) -> tuple[float, float]:
        raise NotImplementedError


class YieldStrategy(VaultRebalancingStrategy):
    '''
    convex optimization of yield
    '''
    def __init__(self, research_engine: ResearchEngine, params: dict):
        '''
        weights excludes base asset balances. Initialize at 0 (all base)
        wealth: base holding = wealth - sum(weights). Initialize at 1.
        '''
        super().__init__(research_engine, params)
        self.progress_display = pd.DataFrame()
        self.current_time = datetime.now()

    def predict(self, index: datetime) -> tuple[np.array, np.array]:
        '''
        uses research engine to predict apy at index. also returns tvl for dilution.
        '''
        # model: sklearn.base.BaseEstimator = list(self.research_engine.fitted_model.values())[0]
        model: sklearn.base.BaseEstimator = self.research_engine.get_model(0)
        return model.predict(index)


    def solve_cvx_problem(self, predicted_apys, tvl, **kwargs):
        # Define the variables and parameters
        N = len(predicted_apys)
        x = cp.Variable(shape=N, nonneg=True, value=deepcopy(self.state.weights/self.state.wealth))
        a = cp.Parameter(shape=N, nonneg=True, value=predicted_apys.clip(min=0.0))
        a_over_tvl = cp.Parameter(shape=(N, N), PSD=True, value=np.diag(predicted_apys/tvl.clip(min=1e-18)*self.state.wealth))
        x0 = cp.Parameter(shape=N, nonneg=True, value=self.state.weights/self.state.wealth)
        # cost_optimization plays in the objective, BUT NOT in update_wealth
        cost = cp.Parameter(shape=N, nonneg=True, value=self.cost_optimization)
        max_x = cp.Parameter(shape=N, nonneg=True, value=[self.parameters['concentration_limit']]*N)
        max_sumx = cp.Parameter(value=(1.0 - self.parameters['base_buffer']))

        # Define the DCP expression. TODO:Trick y to make DPP compliant using y ?
        # assume x << tvl
        objective = cp.Minimize(-a @ x + cost @ cp.abs(x - x0) + cp.quad_form(x, a_over_tvl))
        constraints = [cp.sum(x) <= max_sumx, x <= max_x]

        # Solve the problem and print stdout
        problem = cp.Problem(objective, constraints)
        if kwargs['verbose']:
            with io.StringIO() as buf, redirect_stdout(buf):
                try:
                    problem.solve(**kwargs)
                except cvxerror.SolverError as e:
                    logging.getLogger('defillama').warning(buf.getvalue())

        else:
            problem.solve(**kwargs)


        if problem.status == 'optimal':
            assert np.sum(x.value) < 1.0, "negative base holding; maybe try a larger buffer than {}".format(self.parameters['base_buffer'])

        return {'success': problem.status,
                'y': problem.value if problem.status == 'optimal' else None,
                'x': x.value * self.state.wealth if problem.status == 'optimal' else None}

    def optimal_weights(self, predicted_apys: np.array, tvl: np.array) -> np.array:
        '''
        fit_predict on current features -> predicted_apy -> convex optimization under wealth constraint + cost/risk penalty
        '''
        ####### CVXPY version #######
        res = self.solve_cvx_problem(predicted_apys, tvl,  **self.parameters['solver_params'])
        if hasattr(self, 'gas_reduction_routine'):
            new_res = self.state.weights + self.gas_reduction_routine(predicted_apys, res['x']-self.state.weights)
        else:
            new_res = res['x']

        # brutally trim if solver didn't
        base_weight = self.state.wealth - np.sum(new_res)
        if base_weight < self.state.wealth * self.parameters['base_buffer']:
            logger = logging.getLogger('defillama')
            logger.warning("base_weight {} < {}".format(base_weight/self.state.wealth, self.parameters['base_buffer']))

        return new_res

    def update_wealth(self, new_weights: np.array, prev_state: State, prev_index: datetime,
                      cur_performance: pd.Series) -> dict:
        """
        update wealth from yields, tx cost, and gas
        """
        new_base_weight = self.state.wealth - np.sum(new_weights)

        # yields * dt
        dt = (cur_performance.name - prev_index) / timedelta(days=365)
        tvl = self.research_engine.X.xs(level=('feature', 'window'), key=('tvl', 'as_is'), axis=1).loc[prev_index].ffill().fillna(0.0).values.clip(min=1e-18)
        dilutor = 1 / (1 + prev_state.weights/tvl)
        yields_dt = cur_performance.values * dilutor * dt
        base_yield_dt = 0.0

        # costs, evaluate before strategy value move
        transaction_costs = np.dot(self.cost_wealth, np.abs(new_weights - self.state.weights))
        gas = sum([self.parameters['gas'] if abs(new_weights[i] - self.state.weights[i]) > 1e-8 else 0
                   for i in range(len(new_weights))])

        # strategies and vault pnl after yield accrual
        self.state.weights = new_weights * np.exp(yields_dt)
        new_base_weight *= np.exp(base_yield_dt)
        self.state.wealth = np.sum(self.state.weights) + new_base_weight - transaction_costs - gas

        return {'transaction_costs': transaction_costs,
                'gas': gas,
                'dilutor': dilutor}
#
#     def solve_scipy_problem(self, predicted_apys: np.array):
#         ### objective is APY - txcost (- risk?)
#         # TODO could easily add vol penalty
#         # TODO could easily discount apy by our mktimpact (linear dilution of apy)
#         objective = lambda x: -(
#             np.dot(x, predicted_apys)
#             #                - self.transaction_cost(self.state.weights, x)
#         )
#         objective_jac = lambda x: -(
#             predicted_apys
#             #               - self.transaction_cost.jacobian(self.state.weights, x)
#         )
#         constraints = [{'type': 'ineq',
#                         'fun': lambda x: self.state.wealth * (1.0 - self.parameters['base_buffer']) - np.sum(x),
#                         'jac': lambda x: -np.ones(self.features.shape[1])}]
#         bounds = scipy.optimize.Bounds(lb=np.zeros(self.features.shape[1]),
#                             ub=np.full(self.features.shape[1], self.state.wealth))
#
#         # --------- verbose callback function: breaks down pnl during optimization
#         def callbackF(x, details=None, print_with_flag=None):
#             new_progress = pd.Series({
#                                          'predicted_apy': np.dot(x, predicted_apys) / max(1e-8, np.sum(x)),
#                                          'tx_cost': self.transaction_cost(self.state.weights, x),
#                                          'wealth_constraint (=base weight)': constraints[0]['fun'](x),
#                                          'status': print_with_flag,
#                                          'jac_error': scipy.optimize.check_grad(objective, objective_jac, x),
#                                      }
#                                      | {f'weight_{i}': value for i, value in enumerate(x)}
#                                      | {f'pred_apy_{i}': value for i, value in enumerate(predicted_apys)})
#                                 #TODO:| {f'spot_apy_{i}': value for i, value in enumerate(history.iloc[-1].values)})
#             pfoptimizer_path = os.path.join(os.sep, os.getcwd(), "logs")
#             if not os.path.exists(pfoptimizer_path):
#                 os.umask(0)
#                 os.makedirs(pfoptimizer_path, mode=0o777)
#             pfoptimizer_filename = os.path.join(os.sep, pfoptimizer_path, "{}_optimization.csv".format(
#                 self.current_time.strftime("%Y%m%d-%H%M%S")))
#             self.progress_display = pd.concat([self.progress_display, new_progress], axis=1)
#             self.progress_display.T.to_csv(pfoptimizer_filename, mode='w')
#
#         if 'warm_start' in self.parameters and self.parameters['warm_start']:
#             x1 = self.state.weights
#         else:
#             x1 = np.zeros(self.features.shape[1])
#         if 'verbose' in self.parameters and self.parameters['verbose']:
#             callbackF(x1, details=None, print_with_flag='initial')
#         ftol = self.parameters['ftol'] if 'ftol' in self.parameters else 1e-4
#         finite_diff_rel_step = self.parameters[
#             'finite_diff_rel_step'] if 'finite_diff_rel_step' in self.parameters else 1e-2
#         method = 'SLSQP'
#         if method == 'SLSQP':
#             res = scipy.optimize.minimize(objective, x1, method=method, jac=objective_jac,
#                                constraints=constraints,  # ,loss_tolerance_constraint
#                                bounds=bounds,
#                                callback=(lambda x: callbackF(x, details=None, print_with_flag='interim')) if (
#                                            'verbose' in self.parameters and self.parameters['verbose']) else None,
#                                options={'ftol': ftol, 'disp': False, 'finite_diff_rel_step': finite_diff_rel_step,
#                                         'maxiter': 50 * self.features.shape[1]})
#         elif method == 'trust-constr':
#             res = scipy.optimize.minimize(objective, x1, method=method, jac=objective_jac,
#                                constraints=[scipy.optimize.LinearConstraint(np.array([np.ones(self.features.shape[1])]),
#                                                              [0],
#                                                              [self.state.wealth * (
#                                                                          1 - self.parameters['base_buffer'])])],
#                                # ,loss_tolerance_constraint
#                                bounds=bounds,
#                                callback=(
#                                    lambda x, details: callbackF(x, details=details, print_with_flag='interim')) if (
#                                        'verbose' in self.parameters and self.parameters['verbose']) else None,
#                                options={'gtol': ftol, 'disp': False,
#                                         'maxiter': 50 * self.features.shape[1]})
#         if not res['success']:
#             # cheeky ignore that exception:
#             # https://github.com/scipy/scipy/issues/3056 -> SLSQP is unfomfortable with numerical jacobian when solution is on bounds, but in fact does converge.
#             violation = - min([constraint['fun'](res['x']) for constraint in constraints])
#             if res['message'] == 'Iteration limit reached':
#                 logging.getLogger('pfoptimizer').warning(res[
#                                                              'message'] + '...but SLSQP is uncomfortable with numerical jacobian when solution is on bounds, but in fact does converge.')
#             elif res['message'] == "Inequality constraints incompatible" and violation < self.state.wealth / 100:
#                 logging.getLogger('pfoptimizer').warning(res['message'] + '...but only by' + str(violation))
#             else:
#                 logging.getLogger('pfoptimizer').critical(res['message'])
#         if 'verbose' in self.parameters and self.parameters['verbose']:
#             callbackF(res['x'], details=None, print_with_flag=res['message'])
#         return res

    def gas_reduction_routine(self, predicted_APY, weights_chg, **kwargs):
        #### algo to reduce nb tx after optimization, while keeping sum(weights) invariant
        # order expected_dollar_chg by descending expected_dollar_chg
        assert (predicted_APY >= 0).all(), "gas_reduction_routine assumes positive APY"

        new_chg = copy.deepcopy(weights_chg)
        expected_dollar_chg = predicted_APY * weights_chg * self.research_engine.label_map['apy']['horizons'][0] / 365
        ordering_by_chg = sorted(range(len(expected_dollar_chg)), key=lambda i: expected_dollar_chg[i], reverse=True)
        chg_of_chg = 0
        for i in ordering_by_chg:
            # cut small deposits. This will run first as we order by descending expected_dollar_chg
            if 0 <= expected_dollar_chg[i] < 2 * self.parameters['gas']:
                new_chg[i] = 0
                chg_of_chg += weights_chg[i]
            elif expected_dollar_chg[i] < 0:
                # cut small withdrawals until chg_of_chg is consumed
                if chg_of_chg >= -weights_chg[i]:
                    # if there is still some chg_of_chg to consume, consume it
                    new_chg[i] = 0
                    chg_of_chg += weights_chg[i]
                else:
                    # when we run, consume the stub and break
                    new_chg[i] = weights_chg[i] + chg_of_chg
                    chg_of_chg = 0
                    break
            # note that chg_of_chg may not be exhausted. fine.

        logger = logging.getLogger('defillama')
        logger.info('{}tx, reduced from {}'.format(
            sum(abs(n) > 1e-6 for n in new_chg),
            sum(abs(n) > 1e-6 for n in weights_chg)))
        return new_chg
#
# class TransactionCost:
#     '''
#     UNUNSED by cvxpy
#     cost to switch from a strategy to another. with optional context kwargs.
#     '''
#
#     def __init__(self, params: dict):
#         self.params = params
#
#     def __call__(self, state_0: np.array, state_1: np.array, **kwargs):
#         raise NotImplementedError
#
#     def jacobian(self, state_0: np.array, state_1: np.array, **kwargs):
#         raise NotImplementedError
#
# class HardCodedTransactionCost(TransactionCost):
#     '''
#     UNUNSED by cvxpy
#     described by a pos symetric n x n matrix with 0 diagonal
#     '''
#
#     def __init__(self, params: dict):
#         super().__init__(params)
#         matrix = self.params['cost_matrix']
#         assert matrix.shape[0] == matrix.shape[1], "need square cost matrix"
#         assert matrix.transpose() == matrix, "need symetric cost matrix"
#         assert all(matrix[i, i] == 0 for i in range(matrix.shape[0])), "need zero diagonal"
#         assert all(matrix[i, 0] >= 0 for i in range(matrix.shape[0])), "cost is positive"
#
#     def __call__(self, state_0: np.array, state_1: np.array, **kwargs):
#         assert len(state_0) == self.params['cost_matrix'].shape[0], "cost matrix vs State mismatch"
#         matrix = self.params['cost_matrix']
#         raise NotImplementedError
#
# class TransactionCostThroughBase(TransactionCost):
#     '''
#     UNUNSED by cvxpy
#     we trade through base (WETH) so only a n-1 vector is supplied as cost_vector
#     TODO: please note that this it does as if slippage was paid in base, which is only true when we sell for swaps
#     '''
#
#     def __call__(self, state_0: np.array, state_1: np.array, **kwargs):
#         assert len(state_0) == len(self.params['cost_vector']), "cost matrix vs State mismatch"
#         return np.dot(self.params['cost_vector'], np.abs(state_1 - state_0))
#
#     def jacobian(self, state_0: np.array, state_1: np.array, **kwargs):
#         assert len(state_0) == len(self.params['cost_vector']), "cost matrix vs State mismatch"
#         return np.array(
#             [cost * (1 if state_1[i] > state_0[i] else -1) for i, cost in enumerate(self.params['cost_vector'])])
#
