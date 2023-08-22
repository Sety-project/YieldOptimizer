from itertools import product
import sys
import json
import os
from pathlib import Path
import pandas as pd
import pickle

import yaml

from research.research_engine import build_ResearchEngine, TrivialEwmPredictor, model_analysis
from strategies.vault_rebalancing import YieldStrategy
from strategies.vault_backtest import VaultBacktestEngine
from strategies.cta_strategy import SingleAssetStrategy
from strategies.cta_backtest import BacktestEngine
from utils.api_utils import extract_args_kwargs
from copy import deepcopy

if __name__ == "__main__":
    args, kwargs = extract_args_kwargs(sys.argv)
    if args[0] == 'vault':
        # load parameters from yaml
        with open(args[1], 'r') as fp:
            parameters = yaml.safe_load(fp)

        parameter_grid = {"cap": [0.3],
                          "haflife": ["10d"],
                          "cost": [0.0001, 0.0005, 0.001, 0.005],
                          "gaz": [0.1, 40],
                          "assumed_holding_days": [1, 10, 30],
                          "base_buffer": [0.0, 0.1, .25],
                          "concentration_limit": [0.5, 0.7]}

        def modify_target_with_argument(target: dict, argument: dict) -> dict:
            result = deepcopy(target)
            if "cap" in argument:
                result["run_parameters"]["models"]["haircut_apy"]["TrivialEwmPredictor"]["params"]['cap'] = argument['cap']
            if "haflife" in argument:
                result["run_parameters"]["models"]["haircut_apy"]["TrivialEwmPredictor"]["params"]['halflife'] = argument['haflife']
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

        def dict_list_to_combinations(d: dict) -> list:
            keys = d.keys()
            values = d.values()
            combinations = [dict(zip(keys, combination)) for combination in product(*values)]
            return combinations

        # data
        engine = build_ResearchEngine(parameters)
        performance = pd.concat(engine.performance, axis=1)

        result: list[dict] = list()
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

        pd.DataFrame(result).to_csv(
            os.path.join(os.sep, os.getcwd(), "logs", 'grid.csv'))

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