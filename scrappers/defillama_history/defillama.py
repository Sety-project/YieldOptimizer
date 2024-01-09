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
    pegged_symbols = {'usd': ['usdt', 'usdc', 'dai', 'frax', 'lusd', 'mim', 'susd', 'fraxbp', 'mkusd'],
                      'eth': ['weth', 'eth', 'steth', 'wsteth', 'reth', 'reth2', 'frxeth', 'sfrxeth', 'sweth', 'cbeth',
                              'oeth'],
                      'btc': ['wbtc']}
    def __init__(self, reference_asset: str, chains: list[str], oracle,
                 database: str,
                 use_oracle: bool = False):
        self.logger = logging.getLogger('defillama_history')
        super().__init__()
        self.chains = chains
        self.oracle = oracle
        self.use_oracle = use_oracle
        self.reference_asset = reference_asset
        self.sql_api: SqlApi = SqlApi(st.secrets[database])
        self.connection: Connection = self.sql_api.engine.connect()

        self.protocols = self.get_protocols()
        self.shortlisted_tokens = None
        self.pools = self.get_pools_yields()

    def filter(self, underlyings: list,
               protocol_filters: dict,
               pool_filters: dict) -> None:
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
        protocols = self.protocols

        # normalize
        protocols['name'] = protocols['name'].apply(lambda s: s.lower().replace(' ', '-'))

        # filter
        if self.chains:
            protocols = protocols[protocols['chains'].apply(lambda x: any(y in self.chains for y in x))]
        if 'tvl' in kwargs:
            protocols = protocols[(protocols['tvl'] >= kwargs['tvl']) | (protocols['name'] == 'merkl')]
        if 'listedAt' in kwargs:
            protocols = protocols[(protocols['listedAt'].fillna(1e11).apply(date.fromtimestamp) <= kwargs['listedAt'])]
        if 'category' in kwargs:
            protocols = protocols[~protocols['category'].isin(kwargs['category'])]
        if selected_protocols:
            protocols = protocols[protocols['name'].isin(selected_protocols)]

        return protocols

    def filter_pools(self, **kwargs) -> None:
        # fetch
        pools = self.pools

        # normalize
        pools['project'] = pools['project'].apply(lambda s: s.lower().replace(' ', '-'))
        pools['underlyingTokens'] = pools['underlyingTokens'].apply(lambda s: [x.lower() for x in s] if s else None)
        pools['rewardTokens'] = pools['rewardTokens'].apply(lambda s: [x.lower() for x in s] if s else None)
        pools['name'] = pools.apply(lambda x: '_'.join([x[key]
                                                        for key in ['chain', 'project', 'symbol', 'poolMeta']
                                                        if x[key]])[:63],
                                    axis=1)

        # filter
        pools = pools[pools['chain'].isin(self.chains)]
        pools = pools[pools['tvlUsd'] >= kwargs['tvlUsd']]
        pools = pools[pools['project'].isin(self.protocols['name'].values)]
        pools = pools[pools.apply(lambda x: all(y in self.shortlisted_tokens[x['chain']].values
                                                for y in x['underlyingTokens'])
        if x['underlyingTokens'] is not None else False,
                                  axis=1)]

        return pools

    #@ignore_error
    #@cache_data
    async def apy_history(self, metadata: dict,  **kwargs) -> pd.DataFrame:
        '''gets various components of apy history from defillama
        DB errors may be solved by purging queries at https://console.aiven.io/account/xxxxxxx/project/streamlit/services/streamlit/current-queries'''

        # caching #TODO: daily for now
        last_updated = await self.sql_api.last_updated(metadata)
        if (last_updated is not None
            and last_updated > datetime.now(
                    timezone.utc) - timedelta(days=1)):
            kwargs['fetch_summary'][metadata["name"]] = 'from db'
            return await self.sql_api.read(metadata['name'])

        # get pool history
        try:
            pool_history = await async_wrap(self.get_pool_hist_apy)(metadata['pool'])
        except Exception as e:
            self.logger.warning(f'{metadata["pool"]} {str(e)}')
            kwargs['fetch_summary'][metadata["name"]] = 'error'
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
        kwargs['fetch_summary'][metadata["name"]] = fetch_message
        return result.set_index('date')

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

    def all_apy_history(self, **kwargs) -> FileData:
        metadata = [x.to_dict() for _, x in self.pools.iterrows()]
        coros = [self.apy_history(meta, **kwargs) for meta in metadata] + [self.sql_api.write_metadata(pd.DataFrame(metadata))]
        data = asyncio.run(safe_gather(coros))

        return FileData({key['name']: value for key, value in zip(metadata, data[:-1])})


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