import sys
import json
import os
from pathlib import Path
import pandas as pd
import pickle

import yaml

from research.research_engine import build_ResearchEngine, model_analysis
from strategies.vault_backtest import VaultBacktestEngine
from strategies.cta_strategy import SingleAssetStrategy
from strategies.cta_backtest import BacktestEngine
from utils.api_utils import extract_args_kwargs


if __name__ == "__main__":
    args, kwargs = extract_args_kwargs(sys.argv)
    if args[0] == 'vault':
        # load parameters from yaml
        with open(args[1], 'r') as fp:
            parameters = yaml.safe_load(fp)

        parameter_grid = {"cap": [0.3],
                          "haflife": ["10d"],
                          "cost_blind_optimization": [False, True],
                          "cost": [0.0001, 0.0005, 0.001, 0.005],
                          "gaz": [0.1, 40],
                          "assumed_holding_days": [10, 30],
                          "base_buffer": [0.0, 0.1],
                          "concentration_limit": [0.4, 0.8]}

        result = VaultBacktestEngine.run_grid(parameter_grid, parameters)

        try:
            result.to_csv(
                os.path.join(os.sep, os.getcwd(), "logs", 'grid.csv'))
        except Exception as e:
            result.to_csv(
                os.path.join(os.sep, os.getcwd(), "logs", 'grid2.csv'))

    elif args[0] == 'cta':
        # load parameters
        with open(args[0], 'r') as fp:
            parameters = json.load(fp)

        '''
        model caching for big ML:
            if pickle_override is set, just load pickle and skip
            else run research and print pickle_override if specified
        '''
        outputdir = os.path.join(os.sep, Path.home(), "mktdata", "binance", "results")
        skip_run_research = False
        if "pickle_override" in parameters['input_data']:
            outputfile = os.path.join(os.sep, outputdir, parameters['input_data']["pickle_override"])

            try:
                with open(outputfile, 'rb') as fp:
                    engine = pickle.load(fp)
                    print(f'read {outputfile}')
                    # then fit won't pickle
                    parameters['run_parameters']['pickle_output'] = None
                    skip_run_research = True
            except FileNotFoundError:
                parameters['run_parameters']['pickle_output'] = parameters['input_data']["pickle_override"]

        if not skip_run_research:
            engine = build_ResearchEngine(parameters)

        # backtest
        delta_strategy = SingleAssetStrategy(parameters['strategy'], engine)
        backtest = BacktestEngine(parameters['backtest'], delta_strategy)
        result = backtest.backtest_all(engine)
        result.to_csv(os.path.join(os.sep, outputdir, 'backtest_trajectories.csv'))
        print(f'backtest')

        # analyse perf
        analysis = backtest.perf_analysis(os.path.join(os.sep, outputdir, 'backtest_trajectories.csv'))
        analysis.to_csv(os.path.join(os.sep, outputdir, 'perf_analysis.csv'))
        print(f'analyse perf')

        # inspect model
        model = model_analysis(engine)
        model.to_csv(os.path.join(os.sep, outputdir, 'model_inspection.csv'))
        print(f'inspect model')