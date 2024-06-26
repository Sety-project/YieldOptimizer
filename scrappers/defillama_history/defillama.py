#!/usr/bin/env python
import asyncio
import functools
import logging
import os
import sys
from copy import deepcopy
from typing import Callable

import streamlit as st
import yaml

from research.research_engine import FileData
from scrappers.defillama_history.coingecko import myCoinGeckoAPI
from utils.postgres import SqlApi, Connection
from utils.streamlit_utils import MyProgressBar

try:
    from utils.async_utils import async_wrap, safe_gather
except ImportError:
    sys.path.append(os.getcwd())    # needed when run from project root directory
    from utils.async_utils import async_wrap, safe_gather

from utils.io_utils import ignore_error
from datetime import datetime, timedelta, timezone, date
from abc import abstractmethod
import pandas as pd
from defillama2 import DefiLlama

import requests
import json


class DefiLlamaLiveScrapper(DefiLlama):
    '''this takes a defillama snapshot that can be used as market data for vault rebalancing
    it's a backup if we don't manage to implement contract adapators in time'''
    def __init__(self,
                 filename: str,
                 pool_filter: Callable = (lambda x: True)):
        super().__init__()
        self.filename = filename
        # pool filter is a function that takes a row of the pool dataframe and returns a boolean
        self.pool_filter = pool_filter

    async def snapshot(self):
        pools = await async_wrap(self.get_pools_yields)()
        pools = pools[pools.apply(self.pool_filter, axis=1)]
        timestamp = datetime.now(tz=None)
        # block_heights_list = await safe_gather([async_wrap(defillama.get_closest_block)(chain, timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        #          for chain in pools['chain'].unique()])
        # block_heights = {chain: result for chain, result in zip(pools['chain'].unique(), block_heights_list)}
        pools['local_timestamp'] = timestamp
        #pools['block'] = pools['chain'].apply(lambda x: block_heights[x])
        # TODO: Hatim you need to send a kafka
        pools.to_csv(self.filename, mode='a', header=not os.path.isfile(self.filename))
        logging.getLogger('defillama_scrapper').info(f'snapshot taken to {self.filename}')
        return pools

    async def start(self, frequency=timedelta(hours=1)):
        '''only used for local testing'''
        while True:
            try:
                cur_time = datetime.now(tz=None)
                await self.snapshot()
            except Exception as e:
                self.logger.error(e)
            finally:
                await asyncio.sleep((cur_time + frequency - datetime.now(tz=None)).total_seconds())

def cache_decorator(func):
    @functools.wraps(func)
    @st.cache_data
    def wrapper_cache_decorator(_self, *args, **kwargs):
        return func(_self, *args, **kwargs)
    return wrapper_cache_decorator
def apply_decorator(cls):
    for name, method in DefiLlama.__dict__.items():
        if callable(method) and "get_" in name:
            setattr(cls, name, cache_decorator(method))
    return cls

#@apply_decorator
class FilteredDefiLlama(DefiLlama):
    '''
    filters protocols and pools from defillama
    '''
    pegged_symbols = {'usd': ['usdt', 'usdc', 'dai', 'frax', 'lusd', 'mim', 'susd', 'fraxbp', 'mkusd', 'dola', 'usdc.e', 'susde', 'usdbc', 'sexydai'],
                      'eth': ['weth', 'eth', 'steth', 'wsteth', 'reth', 'reth2', 'frxeth', 'sfrxeth', 'sweth', 'cbeth',
                              'oeth','oseth', 'ezeth', 'weeth', 'apxeth', 'pteeth', 'ptezeth', 'ptezeth'],
                      'btc': ['wbtc']}

    def __init__(self, reference_asset: str, chains: list[str], oracle,
                 database: SqlApi,
                 use_oracle: bool = False,
                 ):
        self.logger = logging.getLogger('defillama_history')
        super().__init__()
        self.chains = chains
        self.oracle = oracle
        self.use_oracle = use_oracle
        self.reference_asset = reference_asset
        self.sql_api: SqlApi = database
        self.connection: Connection = self.sql_api.engine.connect()

        self.all_protocol = self.get_protocols()
        self.protocols: pd.DataFrame = self.all_protocol
        self.shortlisted_tokens = None
        self.all_pool = self.get_pools_yields()
        self.pools: pd.DataFrame = self.all_pool

    def filter(self, underlyings: list,
               protocol_filters: dict,
               pool_filters: dict,
               chains: list = None) -> None:
        if chains:
            self.chains = chains
        self.protocols = self.filter_protocols(**protocol_filters)
        self.shortlisted_tokens = self.filter_underlyings(underlyings)
        self.pools = self.filter_pools(**pool_filters)

    def filter_underlyings(self, pegged_symbols: list) -> None:
        '''filter underlyings'''
        table = self.oracle.address_map.set_index('symbol')
        # keep only symbols that are pegged
        return table.loc[table.index.isin(pegged_symbols), self.chains].dropna(how='all', axis=0)

    def filter_protocols(self,
                         selected_protocols: list = None,
                         **kwargs) -> None:
        # fetch
        protocols = self.all_protocol

        # normalize
        protocols['name'] = protocols['name'].apply(lambda s: s.lower().replace(' ', '-'))

        # filter
        if self.chains:
            protocols = protocols[protocols['chains'].apply(lambda x: any(y in self.chains for y in x))]
        if 'tvl' in kwargs:
            protocols = protocols[(protocols['tvl'] >= kwargs['tvl']) | (protocols['name'] == 'merkl')]
        if 'listedAt' in kwargs:
            protocols = protocols[(protocols['listedAt'].fillna(1e11).apply(date.fromtimestamp) <= kwargs['listedAt'])]
        if 'categories' in kwargs:
            protocols = protocols[protocols['category'].isin(kwargs['categories'])]
        if selected_protocols is not None:
            protocols = protocols[protocols['name'].isin(selected_protocols)]

        return protocols

    def filter_pools(self, **kwargs) -> None:
        # fetch
        pools = self.all_pool

        # normalize
        pools['project'] = pools['project'].apply(lambda s: s.lower().replace(' ', '-'))
        pools['underlyingTokens'] = pools['underlyingTokens'].apply(lambda s: [x.lower() for x in s if x] if s else None)
        pools['rewardTokens'] = pools['rewardTokens'].apply(lambda s: [x.lower() for x in s if x if x] if s else None)
        pools['name'] = pools.apply(lambda x: '_'.join([x[key]
                                                        for key in ['chain', 'project', 'symbol', 'poolMeta']
                                                        if x[key]])[:63],
                                    axis=1)

        # filter
        pools = pools[pools['chain'].isin(self.chains)]
        pools = pools[pools['tvlUsd'] >= kwargs['tvlUsd']]
        pools = pools[pools['apy'] >= kwargs['apy']]
        pools = pools[pools['apyMean30d'] >= kwargs['apyMean30d']]
        pools = pools[pools['project'].isin(self.protocols['name'].values)]
        pools = pools[pools.apply(lambda x: all(y in self.shortlisted_tokens[x['chain']].values
                                                for y in x['underlyingTokens'])
        if x['underlyingTokens'] is not None else False,
                                  axis=1)]

        return pools

    #@ignore_error
    #@cache_data
    async def apy_history(self,
                          metadata: dict,
                          fetch_summary: dict, # fetch_summary[metadata["name"]] = (message, updated_time)
                          progress_bar: MyProgressBar,
                          populate_db: bool = False,
                          **kwargs) -> pd.DataFrame:
        '''gets various components of apy history from defillama
        - if populate_db:
            - if available: fetch from db
            - if not: fetch from defillama and write to db
        - if not populate_db:
            - if available: fetch from db
            - if not: return empty dataframe
        DB errors may be solved by purging queries at https://console.aiven.io/account/xxxxxxx/project/streamlit/services/streamlit/current-queries'''

        # caching #TODO: daily for now
        last_updated = await self.sql_api.last_updated(metadata['name'])
        if (not populate_db) or (last_updated > datetime.now(
                    timezone.utc) - timedelta(days=1)):
            fetch_summary[metadata["name"]] = ('from db', last_updated)
            progress_bar.increment(text=f'From db: {metadata["name"]}')
            return await self.sql_api.read_one(metadata['name'])

        # get pool history
        try:
            pool_history = await async_wrap(self.get_pool_hist_apy)(metadata['pool'])
        except Exception as e:
            self.logger.warning(f'{metadata["pool"]} {str(e)}')
            fetch_summary[metadata["name"]] = ('error', None)
            progress_bar.increment(text=f'Error: {metadata["name"]}: {str(e)}')
            return pd.DataFrame()
        if self.use_oracle:
            pool_history = await self.fetch_oracle(metadata, pool_history)

        apy = pool_history['apy']
        apyReward = pool_history['apyReward']

        haircut_apy = apy
        if kwargs is not None and 'reward_history' in kwargs and metadata['rewardTokens'] not in [None, []]:
            token_addrs_n_chains = {rewardToken: metadata['chain'] for rewardToken in metadata['rewardTokens']}
            reward_discount = await self.discount_reward_by_minmax(token_addrs_n_chains, **kwargs)
            interpolated_discount = \
                        reward_discount.reindex(apyReward.index.union(reward_discount.index)).interpolate().loc[apyReward.index]
            apyReward = apyReward.interpolate(method='linear').fillna(0)
            haircut_apy = apy - (1 - interpolated_discount) * apyReward

        # pathetic attempt at IL...
        il = pool_history['il7d'].fillna(0) * 52
        tvl = pool_history['tvlUsd']


        res_dict = {'haircut_apy': haircut_apy / 100,
                               'apy': apy / 100,
                               'apyReward': apyReward / 100,
                               'il': il,
                               'tvl': tvl}
        if metadata['underlyingTokens'] is None and hasattr(self, 'oracle'):
            assert len(metadata['underlyingTokens']) < 7, 'limited to 6 underlyings for now'
            res_dict |= {f'underlying{i}': pool_history[f'underlying{i}']
                         for i, _ in enumerate(metadata['underlyingTokens'])}

        result = pd.DataFrame(res_dict).reset_index().rename(columns={'index': 'date'})
        result['date'] = result['date'].apply(lambda t: pd.to_datetime(t, unit='ns', utc=True))
        fetch_message = await self.sql_api.write(result, metadata['name'])
        fetch_summary[metadata["name"]] = (fetch_message, datetime.now(timezone.utc))
        progress_bar.increment(text=fetch_message)
        return result.set_index('date')

    def refresh_apy_history(self, progress_bar: MyProgressBar, populate_db: bool = False,) -> tuple[FileData, dict]:
        if not populate_db:
            list_table = self.sql_api.list_tables()
            fetch_summary = {name: ('skipped (not in db)', None)
                             for name in self.pools['name'] if name not in list_table}
            self.pools = self.pools[self.pools['name'].isin(list_table)]
        else:
            fetch_summary = {}
        metadata = [x.to_dict() for _, x in self.pools.iterrows()]
        coros = [self.apy_history(meta,
                                  fetch_summary=fetch_summary,
                                  progress_bar=progress_bar,
                                  populate_db=populate_db) for meta in metadata]
        data = asyncio.run(safe_gather(coros, n=st.session_state.parameters['input_data']['async']['gather_limit']))
        # remove failed pools and update updated_time
        self.pools = self.pools[self.pools['name'].apply(lambda pool: fetch_summary[pool][0] != 'error')]
        self.pools['updated'] = self.pools['name'].apply(lambda pool: fetch_summary[pool][1])
        return FileData({key['name']: value for key, value in zip(metadata, data) if not value.empty}), fetch_summary

    async def fetch_oracle(self, metadata, pool_history):
        if metadata['underlyingTokens'] is not None:
            results = await safe_gather([async_wrap(self.oracle.fetch_market_chart)(
                _id=self.oracle.address_to_id(address, metadata['chain']),
                days='max',
                vs_currency=self.reference_asset)
                for address in metadata['underlyingTokens']])

            for i, (underlyingToken, prices) in enumerate(zip(metadata['underlyingTokens'], results)):
                prices = prices.bfill().set_index('timestamp')
                pool_history = \
                pool_history.join(prices.rename(columns={'price': f'underlying{i}'}), how='outer').interpolate(
                    'index').loc[pool_history.index]
        return pool_history

    @ignore_error
    async def discount_reward_by_minmax(self, token_addrs_n_chains, **kwargs) -> pd.Series:
        kwargs_reward_history = deepcopy(kwargs['reward_history'])
        discount_lookback = kwargs_reward_history.pop('discount_lookback')

        try:
            def get_price_time_series(token_address, **kwargs_reward_history):
                url = 'https://api.coingecko.com/api/v3/coins/ethereum/contract/{}/market_chart/?vs_currency=usd&days={}'.format(token_address, kwargs_reward_history['span'])
                response = requests.get(url)
                data = json.loads(response.text)
                result = pd.Series(name=token_address,
                                   index=[pd.to_datetime(x[0]*1e6) for x in data['prices']],
                                   data=[x[1] for x in data['prices']])
                return result

            reward_history = pd.concat([get_price_time_series(address, **kwargs_reward_history)
                                                for address in token_addrs_n_chains.keys()],
                                       axis=1).ffill()
        except Exception as e:
            reward_history = self.get_prices_at_regular_intervals(token_addrs_n_chains, **kwargs_reward_history)

        reward_discount = pd.DataFrame()
        for rewardToken in reward_history.columns:
            # need to clean defillama data
            token_history = reward_history[rewardToken].dropna()
            # discount by min/max, so that if discounted(max)=min, discounted(min)=min^2/max<min, etc...
            reward_discount[rewardToken] = token_history.rolling(discount_lookback).apply(
                lambda x: min(x) / max(x))
        # defillama doesnt breakdown rewards so take the min...
        reward_discount = reward_discount.min(axis=1)
        return reward_discount


def compute_moments(apy: dict[str, pd.Series]) -> dict[str, pd.Series]:
    new_apy: dict = {}
    for name, df in apy.items():
        # apply apy cutoff date and cap 200%
        if df is None:
            continue
        df = df[df.index < datetime.now().replace(tzinfo=timezone.utc)]
        df = df.applymap(lambda x: min(x, 200))
        # hack for data gaps...
        # df = df.fillna(method='ffill', limit=1)

        # compute moments
        df = df.ewm(halflife=timedelta(days=7), times=df.index).mean()

        new_apy |= {name: df}
    return new_apy


def get_historical_best_pools(apy: dict[str, pd.Series], max_rank: int, start: datetime, end: datetime) -> pd.DataFrame:
    # compute historical pool rank
    haircut_apy = pd.DataFrame({key: value['haircut_apy'] for key, value in apy.items()})
    df = pd.DataFrame(haircut_apy).fillna(method='ffill')
    df = df[(df.index >= start)&(df.index <= end)]
    best_only = df[df.rank(axis=1, ascending=False)<max_rank]
    return best_only.dropna(how='all', axis=1)


def print_top_pools(new_apy: dict[str,pd.DataFrame], filename: str='defillama_hist.xlsx') -> pd.DataFrame:
    # compute top pools
    verbose_index = defillama.pools[['project', 'symbol', 'pool']].set_index('pool')
    top_pools = pd.DataFrame({tuple([key]+verbose_index.loc[key].to_list()): value.iloc[-1] for key, value in new_apy.items()})
    top_pools = top_pools.T.sort_values(by='haircut_apy',ascending=False)
    # save to excel
    try:
        with pd.ExcelWriter(filename, engine='xlsxwriter', mode='w') as writer:
            top_pools.to_excel(writer, datetime.now().strftime(f"{sys.argv[2]} %d %m %Y %H_%M_%S"))
    except PermissionError:
        print('Please close the excel file')
    return top_pools