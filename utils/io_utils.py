#!/usr/bin/env python3
import asyncio
import hmac
import html
import os

import streamlit as st
from plotly import express as px
from plotly.subplots import make_subplots

from utils.async_utils import async_wrap
import functools
from copy import deepcopy
from itertools import product
import retry
import logging
import json
import pandas as pd
import numpy as np
import collections
#from telegram import Bot
import datetime
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


def check_whitelist():
    async def chat():
        async with bot:
            print(st.session_state.user_tg_handle)
            await bot.send_message(chat_id=st.session_state.user_tg_handle, text='wassup?')
            print(chat_id)
            await bot.send_message(chat_id=chat_id, text=f'{st.session_state.user_tg_handle} logged in at {datetime.datetime.now()}')

    if st.session_state.user_tg_handle in st.secrets.whitelist:
        st.session_state.authentification = "verified"

        # bot = Bot(token=st.secrets.get('bot_token'))  # Replace 'YOUR_BOT_TOKEN' with your bot's token
        # chat_id = st.secrets.get('chat_id')  # Replace 'YOUR_CHAT_ID' with the desired chat ID
        # asyncio.run(chat())
    else:
        st.session_state.authentification = "incorrect"
        st.warning(html.unescape('chat {} to get access'.format('https://t.me/daviddarr')))
    st.session_state.password = ""

def plot_perf(backtest: pd.DataFrame, base_buffer: float, height: int=1000, width: int=1500) -> None:
    # plot results
    backtest['pnl']["tx_cost"].iloc[0] = 0.001
    cum_tx_cost = backtest['pnl']["tx_cost"].cumsum()
    apy = (backtest['pnl']['wealth'] / backtest['pnl']['wealth'].shift(1) - 1) * 365 / 1 * 100
    max_apy = backtest['full_apy'].max(axis=1) * 100 * (1 - base_buffer)
    max_apy.name = 'max_apy'

    subfig = make_subplots(specs=[[{"secondary_y": True}]])

    # create two independent figures with px.line each containing data from multiple columns
    fig = px.line(apy, render_mode="webgl", )
    fig3 = px.line(max_apy, render_mode="webgl")
    fig3.update_traces(line={"dash": 'dot'})
    fig2 = px.line(cum_tx_cost, render_mode="webgl", )
    fig2.update_traces(yaxis="y2")
    subfig.add_traces(fig.data + fig2.data + fig3.data)
    subfig.layout.xaxis.title = "time"
    subfig.layout.yaxis.title = "apy"
    subfig.layout.yaxis.range = [0, 30]
    subfig.layout.yaxis2.title = "tx_cost"
    subfig.layout.height = height
    subfig.layout.width = width

    # recoloring is necessary otherwise lines from fig und fig2 would share each color
    # e.g. Linear-, Log- = blue; Linear+, Log+ = red... we don't want this
    subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
    st.plotly_chart(subfig, use_container_width=True)
    st.write(f'apy mean = {apy.mean()}')
    st.write(f'max apy mean = {max_apy.mean()}')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
