#!/usr/bin/env python
import logging
import numbers
import os
import shutil
import sys
import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Callable

import yaml

from scrappers.defillama_history.coingecko import myCoinGeckoAPI
from utils.async_utils import safe_gather
import streamlit as st

try:
    from utils.async_utils import async_wrap, safe_gather
except ImportError:
    sys.path.append(os.getcwd())    # needed when run from project root directory
    from utils.async_utils import async_wrap, safe_gather

from utils.io_utils import ignore_error, async_to_csv
from datetime import datetime, timedelta, timezone, time
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


class FilteredDefiLlama(DefiLlama):
    '''
    filters protocols and pools from defillama
    '''
    def __init__(self, reference_asset: str, chains: list[str], oracle,
                 protocol_tvl_threshold: float,
                 pool_tvl_threshold: float,
                 use_oracle: bool = False):
        self.logger = logging.getLogger('defillama_history')
        super().__init__()
        self.chains = chains
        self.oracle = oracle
        self.use_oracle = use_oracle
        self.reference_asset = reference_asset
        self.protocols = self.filter_protocols(tvl_threshold=protocol_tvl_threshold)
        self.shortlisted_tokens = self.filter_underlyings()
        self.pools = self.filter_pools(tvlUsd_floor=pool_tvl_threshold)

    @abstractmethod
    def filter_underlyings(self) -> None:
        '''filter underlyings'''
        table = self.oracle.address_map.set_index('symbol')
        pegged_symbols = {'usd': ['usdt', 'usdc', 'dai', 'frax', 'lusd', 'mim', 'susd', 'fraxbp', 'mkusd'],
                             'eth': ['weth', 'eth', 'steth', 'wsteth', 'reth', 'reth2', 'frxeth', 'sfrxeth', 'sweth', 'cbeth', 'oeth'],
                             'btc': ['renbtc', 'tbtc', 'hbtc', 'sbtc', 'obtc', 'wbtc', 'bbtc', 'pbtc', 'ebtc', 'ibtc']}
        # keep only symbols that are pegged
        return table.loc[table.index.isin(pegged_symbols[self.reference_asset]), self.chains].dropna()

    def filter_protocols(self,
                         tvl_threshold: float = None,
                         excluded_categories: list = None,
                         excluded_protocols: list = None) -> None:
        protocols = self.get_protocols()
        '''filter protocols'''
        protocols['name'] = protocols['name'].apply(lambda s: s.lower().replace(' ', '-'))
        protocol_filters = {
            'chains': lambda x:  (self.chains is None) or any(y in self.chains for y in x),
            'name': lambda x: (excluded_protocols is None) or (x not in excluded_protocols),
            #   'audits': lambda x: x in ['1', '2', '3'],
            'category': lambda x: (excluded_categories is None) or (x not in excluded_categories),
            #    'listedAt': lambda x: not x>datetime(2023,3,1).timestamp()*1000, # 1mar23
            'tvl': lambda x: (tvl_threshold is None) or (x > tvl_threshold),
            #   'openSource': lambda x: not x == False,
            #   'audited': lambda x: x in ['1','2','3'],
            #    'wrongLiquidity': lambda x: x==True,
            #    'rugged': lambda x: x==True,
        }
        return protocols[protocols.apply(lambda x: all(v(x[k]) for k, v in protocol_filters.items()), axis=1)]

    @abstractmethod
    def filter_pools(self, **kwargs) -> None:
        pools = self.get_pools_yields()
        pool_filters = {
            'chain': lambda x: x in self.chains,
            'project': lambda x: x.lower() in self.protocols['name'].values,
            'tvlUsd': lambda x: x > kwargs['tvlUsd_floor'],
            'ilRisk': lambda x: not x == 'yes',
            #    'exposure': lambda x: x in ['single', 'multi'], # ignore
        }
        pools = pools[pools.apply(lambda x: all(v(x[k]) for k, v in pool_filters.items()), axis=1)]
        pools = pools[pools.apply(lambda x: all(y in self.shortlisted_tokens[x['chain']].values
                                                for y in x['underlyingTokens'])
        if x['underlyingTokens'] is not None else False,
                                  axis=1)]
        pools['name'] = pools.apply(lambda x: '_'.join([x[key]
                                                                  for key in ['chain', 'project', 'symbol', 'poolMeta']
                                                                  if x[key]]),
                                              axis=1)
        return pools

    #@ignore_error
    #@cache_data
    async def apy_history(self, metadata: dict,  **kwargs) -> pd.DataFrame:
        '''gets various components of apy history from defillama'''

        # get pool history
        if 'dirname' in  kwargs and os.path.isfile(
            os.path.join(os.sep, kwargs['dirname'], f"{metadata['name']}.csv")
        ):
            return self.read_history(kwargs, metadata)

        pool_history = await async_wrap(self.get_pool_hist_apy)(metadata['pool'])
        if self.use_oracle:
            pool_history = await self.fetch_oracle(metadata, pool_history)
        if 'status' in kwargs:
            kwargs['status'].update(label=metadata['name'], state="running", expanded=True)

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
            res_dict |= {f'underlying{i}': pool_history[f'underlying{i}']
                         for i, _ in enumerate(metadata['underlyingTokens'])}
        result = pd.DataFrame(res_dict)

        if 'dirname' in kwargs:
            await self.write_history(kwargs, metadata, result)

        return result

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

    def read_history(self, kwargs, metadata):
        pool_history = pd.read_csv(
            os.path.join(os.sep, kwargs['dirname'], f"{metadata['name']}.csv")
        )
        pool_history['date'] = pd.to_datetime(pool_history['date'])
        return pool_history.set_index('date')

    async def write_history(self, kwargs, metadata, result):
        if not os.path.isdir(kwargs['dirname']):
            os.makedirs(kwargs['dirname'], mode=0o777)
        name = os.path.join(os.sep, kwargs['dirname'], '{}.csv'.format(metadata['name']))
        await async_to_csv(result, name, mode='w', header=True)

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

    def all_apy_history(self, **kwargs) -> dict[str, pd.DataFrame]:
            metadata = [x.to_dict() for _, x in self.pools.iterrows()]
            coros = [self.apy_history(meta, **kwargs) for meta in metadata]
            data = asyncio.run(safe_gather(coros))

            # print pool meta in parent dir
            if 'dirname' in kwargs:
                filename = os.path.join(os.sep, os.getcwd(), 'data', f'{self.__class__.__name__}_pool_metadata.csv')
                pd.DataFrame(metadata).set_index('pool').to_csv(filename, mode='w')

            return {key['pool']: value for key, value in zip(metadata, data)}


def compute_moments(apy: dict[str, pd.Series]) -> dict[str, pd.Series]:
    new_apy: dict = dict()
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
        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            top_pools.to_excel(writer, datetime.now().strftime(f"{sys.argv[2]} %d %m %Y %H_%M_%S"))
    except PermissionError:
        print('Please close the excel file')
    return top_pools


if __name__ == '__main__':
    if sys.argv[1] == 'defillama':
        if os.path.isfile(sys.argv[2]):
            with open(sys.argv[2], 'r') as fp:
                config = yaml.safe_load(fp)

            coingecko = myCoinGeckoAPI()
            coingecko.adapt_address_map_to_defillama()
            defillama = FilteredDefiLlama(reference_asset=config['input_data']['pool_selector']['reference_asset'],
                                          chains=config['input_data']['pool_selector']['chains'],
                                          oracle=coingecko,
                                          protocol_tvl_threshold=float(config['input_data']['pool_selector'][
                                              'protocol_tvl_threshold']),
                                          pool_tvl_threshold=float(config['input_data']['pool_selector'][
                                              'pool_tvl_threshold']),
                                          use_oracle=bool(config['input_data']['pool_selector']['use_oracle']))
        else:
            raise FileNotFoundError(f'{sys.argv[2]} not found')

        dirname = os.path.join(os.sep, os.getcwd(), 'data', 'latest')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        all_history = defillama.all_apy_history(dirname=dirname)