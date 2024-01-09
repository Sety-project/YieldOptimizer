#!/usr/bin/env python3

import collections
import functools
import json
import logging
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import retry

from utils.async_utils import async_wrap
from dateutil.parser import parse
from datetime import datetime, timedelta, date

#from telegram import Bot
# this is for jupyter
# import cufflinks as cf
# cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)

'''
I/O helpers
'''

async def async_read_csv(*args,**kwargs):
    coro = async_wrap(pd.read_csv)
    return await coro(*args,**kwargs)

def to_csv(*args,**kwargs):
    return args[0].to_csv(args[1],**kwargs)

async def async_to_csv(*args,**kwargs):
    coro = async_wrap(to_csv)
    return await coro(*args,**kwargs)

@retry.retry(exceptions=Exception, tries=3, delay=1,backoff=2)
def ignore_error(func):
    @functools.wraps(func)
    async def wrapper_ignore_error(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__.split('.')[0])
            logger.warning(f'{str(func)}({args}{kwargs})\n-> {e}', exc_info=False)
    return wrapper_ignore_error

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.core.generic.NDFrame):
            return obj.to_json()
        if isinstance(obj, collections.deque):
            return list(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

def modify_target_with_argument(target: dict, argument: dict) -> dict:
    result = deepcopy(target)
    for path, value in argument.items():
        path = path.split('.')
        cur = result
        for p in path[:-1]:
            cur = cur[p]
        cur[path[-1]] = value
    return result


def extract_from_paths(target: dict, paths: list) -> dict:
    result = {}
    for path in paths:
        atoms = path.split('.')
        cur = target
        for p in atoms:
            cur = cur[p]
        result[path] = cur
    return result


def dict_list_to_combinations(d: dict) -> list[dict]:
    keys = d.keys()
    values = d.values()
    return [dict(zip(keys, combination)) for combination in product(*values)]


def get_date_or_timedelta(x: str, ref_date: date) -> date:
    try:
        return parse(x).date()
    except ValueError:
        try:
            return ref_date - pd.to_timedelta(x)
        except ValueError:
            raise ValueError(f'Could not parse {x} as date or timedelta')

import time

def profile(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            with open(filename, 'a') as f:
                f.write(f"{func.__name__}, {end_time - start_time:.6f}\n")
            return result
        return wrapper
    return decorator


