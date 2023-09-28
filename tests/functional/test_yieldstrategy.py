"""
YieldStrategy tests for the following methods:
- solve_cvx_problem()
- optimal_weights()
- update_wealth()
- predict()
"""
import sys
import unittest
from unittest import mock
from datetime import datetime
import numpy as np
import pandas as pd
from strategies.vault_betsizing import YieldStrategy, State
from research.research_engine import ResearchEngine, Instrument, FileData, TrivialEwmPredictor


params = {
    'input_data': {
        'ResearchName': 'defillama', 'dirpath': ['data', 'DynLst'],
        'selected_instruments': [
            '6d342d6a-9796-443e-8152-b94e8b6021fc', '77020688-e1f9-443c-9388-e51ace15cc32',
            'ce70e358-6668-498c-9afa-651f770eac27', '42c02674-f7a2-4cb0-be3d-ade268838770',
            '747c1d2a-c668-4682-b9f9-296708a3dd90', 'e378a7c2-6285-4993-9397-87ac9c8adc15',
            'a4b5b995-99e7-4b8f-916d-8940b5627d70', '4cc5df76-f81d-49fe-9e1e-4caa6a8dad0b',
            'c9873dab-0979-478c-b48c-3c3a0c935449', '98463e86-e32c-4998-830c-ccd47dbb3041',
            'd4b3c522-6127-4b89-bedf-83641cdcd2eb', '4fc7d3dc-a6f0-40e6-af53-ab9d6b50b5c0',
            'fc7701fc-e290-4606-a8be-9e4ba6c5f91a', 'e6435aae-cbe9-4d26-ab2c-a4d533db9972']},
    'run_parameters': {
        'normalize': False, 'cross_ewma': False, 'groupby': False,
        'models': {'apy': {'TrivialEwmPredictor': {'params': {'cap': 3, 'halflife': '1s'}}}},
        'pickle_file': 'engine.pickle', 'unit_test': False},
    'feature_map': {'apy': {'as_is': None}},
    'label_map': {'apy': {'horizons': [3]}},
    'strategy': {
        'cost': 0.0001, 'gas': False, 'base_buffer': 0.15, 'concentration_limit': 0.4,
        'initial_wealth': 1000000,
        'solver_params': {'solver': 'ECOS', 'verbose': False, 'warm_start': False, 'max_iters': 100,
                          'abstol': 0.01, 'reltol': 0.01, 'feastol': 0.01}},
    'backtest': {'start_date': '2022-09-01T00:00:00', 'end_date': '2023-09-01T00:00:00'}}


class ResearchEngineMock(ResearchEngine):

    def __init__(self, params):
        super().__init__(**params)

    @staticmethod
    def read_data(dirpath, selected_instruments: list[Instrument], start_date) -> FileData:
        # return FileData()
        pass

    def fit(self):
        pass


class YieldStrategyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        instr = params['input_data']['selected_instruments']
        self.engine = ResearchEngineMock(params)
        self.engine.performance = pd.Series(data=[np.nan for _ in range(len(instr))], index=instr,
                                            name=datetime(2023, 9, 1))

        self.strategy = YieldStrategy(self.engine, params['strategy'])
        self.strategy.cost_wealth = np.full(len(instr), params['strategy']['cost'])

    def test_properly_initiated(self):
        """Test if the objects are properly initiated"""
        # print('self.engine.performance:', self.engine.performance)
        self.assertEqual(self.strategy.state.wealth, 1000000)

    # @unittest.skip
    def test_solve_cvx_problem(self):
        """Test convex solver"""

        predicted_apys = np.array([0., 0., 0., 0.0120251, 0.039, 0., 0.0481991, 0.0341478, 0., 0.,
                                   0., 0., 0., 0.])
        res = self.strategy.solve_cvx_problem(predicted_apys, **params['strategy']['solver_params'])
        self.assertEqual(res['success'], 'optimal')
        self.assertAlmostEqual(res['y'], -0.02624, 5)
        expected = [1.05735227e+01, 1.05735227e+01, 1.05735227e+01, 2.00964987e+01, 3.99826588e+05,
                    1.05735227e+01, 3.99987319e+05, 5.00470465e+04, 1.05735227e+01, 1.05735227e+01,
                    1.05735227e+01, 1.05735227e+01, 1.05735227e+01, 1.05735227e+01]
        [self.assertAlmostEqual(a, b, 3) for a, b in zip(res['x'].tolist(), expected)]

    # @unittest.skip
    @mock.patch.object(YieldStrategy, 'solve_cvx_problem')
    def test_optimal_weights(self, mock_solver):
        predicted_apys = np.array([0., 0., 0., 0.0120251, 0.039, 0., 0.0481991, 0.0341478, 0., 0.,
                                   0., 0., 0., 0.])
        mock_solver.return_value = {'x': np.full(len(params['input_data']['selected_instruments']), 0.1)}
        res = self.strategy.optimal_weights(predicted_apys)
        mock_solver.assert_called_with(predicted_apys, **params['strategy']['solver_params'])

    # @unittest.skip
    @mock.patch.object(ResearchEngine, 'get_model')
    def test_predict(self, mock_predict):
        """Test if ResearchEngine.get_model() is invoked"""
        index = datetime.utcnow()

        # Call the method being tested
        self.strategy.predict(index)

        # Make sure the predict method of ResearchEngine is invoked
        mock_predict.assert_called_with(0)

    # @unittest.skip
    def test_update_wealth(self):
        weights = np.array([
            1.05735227e+01, 1.05735227e+01, 1.05735227e+01, 2.00964987e+01, 3.99826588e+05, 1.05735227e+01,
            3.99987319e+05, 5.00470465e+04, 1.05735227e+01, 1.05735227e+01, 1.05735227e+01, 1.05735227e+01,
            1.05735227e+01, 1.05735227e+01])

        instr = params['input_data']['selected_instruments']
        perf = pd.Series(data=[np.nan for _ in range(len(instr))], index=instr,
                         name=datetime(2023, 9, 2))

        # Check for initial initialization when Datetime index is None
        cost, gas = self.strategy.update_wealth(weights, State(weights=weights, wealth=1000000.0),
                                                None, perf.fillna(0.))
        self.assertEqual((cost, gas), (0., 0.))
        self.assertEqual(self.strategy.state.weights.all(), weights.all())

        # Check with non-None index value
        weights = np.full(len(instr), 0.1)
        perf = pd.Series(data=[0.1 for _ in range(len(instr))], index=instr,
                         name=datetime(2023, 9, 2))
        cost, gas = self.strategy.update_wealth(weights, State(weights=weights, wealth=1000000.0),
                                                datetime(2023, 9, 1), perf.fillna(0.))
        self.assertAlmostEqual(cost, 84.9985, 4)
        self.assertAlmostEqual(gas, 0., 4)
