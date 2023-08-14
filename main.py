import sys
import json
import os
from pathlib import Path
import pickle
from cta.research_engine import build_ResearchEgine, model_analysis
from cta.strategy import Strategy
from cta.backtest import BacktestEngine
from utils.api_utils import extract_args_kwargs

if __name__ == "__main__":
    args, kwargs = extract_args_kwargs(sys.argv)
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
    delta_strategy = Strategy(parameters['strategy'])
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