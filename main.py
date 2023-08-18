import sys
import json
import os
from pathlib import Path
import pandas as pd
import pickle
from research.research_engine import build_ResearchEgine, TrivialEwmPredictor, model_analysis
from strategies.vault_rebalancing import YieldStrategy
from strategies.vault_backtest import VaultBacktestEngine
from strategies.cta_strategy import SingleAssetStrategy
from strategies.backtest import BacktestEngine
from utils.api_utils import extract_args_kwargs
from copy import deepcopy

if __name__ == "__main__":
    args, kwargs = extract_args_kwargs(sys.argv)
    if args[0] == 'vault':
        # load parameters
        with open(args[1], 'r') as fp:
            parameters = json.load(fp)

        print(f'data...\n')
        engine = build_ResearchEgine(parameters)

        # backtest
        print(f'backtest...\n')
        backtest = VaultBacktestEngine(parameters['backtest'])
        data = pd.DataFrame(engine.performance)
        data = data.ffill().dropna()
        vault_rebalancing = YieldStrategy(parameters['strategy'],
                                          features=data,
                                          research_engine=engine)
        backtest.perf_analysis(backtest.run(vault_rebalancing))

        parameter_grid = {"initial_wealth": [100],
                          "haflife": ["7d","10d", "14d","21d"],
                          "cost": [0.001],
                          "assumed_holding_days": [5,7,10,14,21,30]}
        parameter_grid = {"initial_wealth": [100],
                          "haflife": ["10d"],
                          "cost": [0.001],
                          "assumed_holding_days": [10]}

        # create parameters_list as a list of dicts from parameter_grid
        original_parameter = parameters
        parameter_dict = dict()
        for initial_wealth in parameter_grid["initial_wealth"]:
            for haflife in parameter_grid["haflife"]:
                for cost in parameter_grid["cost"]:
                    for assumed_holding_days in parameter_grid["assumed_holding_days"]:
                        new_parameter = deepcopy(original_parameter)
                        new_parameter['strategy']['initial_wealth'] = initial_wealth
                        new_parameter["run_parameters"]["models"]["haircut_apy"]["TrivialEwmPredictor"]["params"]['halflife'] = haflife
                        new_parameter['strategy']['cost'] = cost
                        new_parameter["label_map"]["haircut_apy"]["horizons"] = [assumed_holding_days]

                        name = (initial_wealth, haflife, cost, assumed_holding_days)

                        parameter_dict[name]= new_parameter
        result = dict()
        for name, cur_params in parameter_dict.items():
            engine = build_ResearchEgine(cur_params)
            vault_rebalancing = YieldStrategy(cur_params['strategy'],
                                              features=data,
                                              research_engine=engine)
            result[name] = backtest.perf_analysis(backtest.run(vault_rebalancing))

        pd.DataFrame(columns=pd.MultiIndex.from_tuples(parameter_dict.keys()), data=result).T.to_csv(
            os.path.join(os.sep, os.getcwd(), "scrappers", "defillama_history", 'grid.csv'))

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
            engine = build_ResearchEgine(parameters)

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