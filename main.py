import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import yaml

from research.research_engine import build_ResearchEngine, model_analysis
from strategies.vault_backtest import VaultBacktestEngine
from strategies.cta_betsizing import SingleAssetStrategy
from strategies.cta_backtest import BacktestEngine
from utils.api_utils import extract_args_kwargs


if __name__ == "__main__":
    args, kwargs = extract_args_kwargs(sys.argv)
    if args[0] in ['backtest', 'grid']:
        # load parameters from yaml
        with open(args[1], 'r') as fp:
            parameters = yaml.safe_load(fp)
        vault_name = args[1].split(os.sep)[-1].split('.')[0]
        dirname = os.path.join(os.sep, os.getcwd(), "logs", vault_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        parameter_grid = {"cap": [3, 999],
                          "halflife": ["1s", "3d", "10d", "30d"],
                          "cost": [0.0001, 0.0005, 0.001, 0.005],
                          "gas": [False],
                          "assumed_holding_days": [3, 8, 13, 18, 23, 28, 32, 9999],
                          "base_buffer": [0.15],
                          "concentration_limit": [0.4,
                                                  0.7, 1.0]} if args[0] == 'grid' else dict()
        parameter_grid = {"cap": [3],
                          "halflife": ["10d"],
                          "cost": [0.0005],
                          "gas": [False, 10, 20, 40],
                          "assumed_holding_days": [7, 28],
                          "base_buffer": [0.10],
                          "concentration_limit": [0.4,
                                                  0.7, 1.0]} if args[0] == 'grid' else dict()
        result = VaultBacktestEngine.run_grid(parameter_grid, parameters)

        if args[0] == 'grid':
            try:
                result.to_csv(
                    os.path.join(os.sep, dirname, 'grid.csv'))
            except Exception as e:
                result.to_csv(
                    os.path.join(os.sep, dirname, 'grid2.csv'))

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
        delta_strategy = SingleAssetStrategy(parameters['strategy'])
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