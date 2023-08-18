#!/usr/bin/env python3
from utils.async_utils import async_wrap
import functools
import retry
import logging
import json
import pandas as pd
import numpy as np
import collections
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
