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

from coingecko import myCoinGeckoAPI
from utils.async_utils import safe_gather

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
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger('defillama_history')
        super().__init__()
        if config is not None:
            self.protocols = sum((config['protocols'][f'tier{i+1}'] for i in range(1)), [])
            if config['oracle'] == 'coingecko':
                self.oracle = myCoinGeckoAPI()
            else:
                self.logger.warning('no valid oracle specified (only coingecko is supported)')
        else:
            protocols = self.get_protocols()
            self.protocols = list(protocols['name'].apply(lambda s: s.lower().replace(' ', '-')).unique())

        self.shortlisted_tokens = self.filter_underlyings()
        self.pools = self.filter_pools(self.get_pools_yields())

    @abstractmethod
    def filter_underlyings(self) -> dict:
        '''filter underlyings'''
        raise NotImplementedError

    def wide_filter_protocols(self, protocols: pd.DataFrame) -> pd.DataFrame:
        '''filter protocols'''
        excluded_categories = ['NFT Lending',
                               'NFT Marketplace']
        excluded_protocols = ['uwu-lend',
                              'sturdy',
                              'goldfinch',
                              'ribbon',
                              'idle',
                              'ipor',
                              'notional',
                              'merkl',
                              'asymetrix-protocol',
                              'vesper']
        protocol_filters = {
            #    'chain': lambda x: x in ['Ethereum','Multi-Chain'],
            'name': lambda x: x not in excluded_protocols,
            #    'audits': lambda x: x in [None, '0', '1', '2', '3'],
            'category': lambda x: x not in excluded_categories,
            #    'listedAt': lambda x: not x>datetime(2023,3,1).timestamp()*1000, # 1mar23
            #    'tvl': lambda x: x>10e6,
            'openSource': lambda x: not x == False,
            # 'audited': lambda x: not is in ['1','2','3'],
            #    'wrongLiquidity': lambda x: x==True,
            #    'rugged': lambda x: x==True,
        }
        return protocols[protocols.apply(lambda x: all(v(x[k]) for k, v in protocol_filters.items()), axis=1)]

    @abstractmethod
    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    #@ignore_error
    async def apy_history(self, metadata: dict,  **kwargs) -> pd.DataFrame:
        '''gets various components of apy history from defillama'''

        # get pool history
        if os.path.isfile(os.path.join(os.sep, kwargs['dirname'], '{}.csv'.format(metadata['pool']))):
            return self.read_history(kwargs, metadata)

        pool_history = await async_wrap(defillama.get_pool_hist_apy)(metadata['pool'])
        if hasattr(self, 'oracle'):
            pool_history = await self.fetch_oracle(metadata, pool_history)

        apy = pool_history['apy']
        apyReward = pool_history['apyReward']

        haircut_apy = apy
        if kwargs is not None and 'reward_history' in kwargs:
            if metadata['rewardTokens'] not in [None, []]:
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
        pool_history = pd.read_csv(os.path.join(os.sep, kwargs['dirname'], '{}.csv'.format(metadata['pool'])))
        pool_history['date'] = pd.to_datetime(pool_history['date'])
        return pool_history.set_index('date')

    async def write_history(self, kwargs, metadata, result):
        pool_name = '_'.join([metadata[key] for key in ['chain', 'project', 'symbol', 'poolMeta'] if metadata[key]])
        name = os.path.join(os.sep, kwargs['dirname'], '{}.csv'.format(pool_name))
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

    def find_underlyings(self, pools: pd.DataFrame) -> pd.DataFrame:
        '''find underlying tokens of pools'''
        pools['newUnderlyingTokens'] = pools['underlyingTokens'].apply(
            lambda x: [token for token in x if token.lower() not in self.shortlisted_tokens.values()]
            if x is not None else None)
        return pools

    def all_apy_history(self, **kwargs) -> dict[str, pd.DataFrame]:
            metadata = [x.to_dict() for _, x in self.pools.iterrows()]
            coros = [self.apy_history(meta, **kwargs) for meta in metadata]
            data = asyncio.run(safe_gather(coros))

            # print pool meta in parent dir
            if 'dirname' in kwargs:
                filename = os.path.join(os.sep, os.getcwd(), 'data', f'{self.__class__.__name__}_pool_metadata.csv')
                pd.DataFrame(metadata).set_index('pool').to_csv(filename, mode='w')

            return {key['pool']: value for key, value in zip(metadata, data)}


class DiscoveryDefiLlama(FilteredDefiLlama):
    def __init__(self,
                 apy_floor: float,
                 tvlUsd_floor: float,
                 chains: list[str]):
        self.chains = chains
        self.apy_floor = apy_floor
        self.tvlUsd_floor = tvlUsd_floor

        super().__init__()

    def filter_underlyings(self) -> dict:
        '''
        we don't use that here
        '''
        return dict()
    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in self.chains,
            'project': lambda x: x in self.protocols,
            # 'underlyingTokens': lambda x: all(
            #     token.lower() in self.shortlisted_tokens.values() for token in x),
            'tvlUsd': lambda x: x > self.tvlUsd_floor,
            'ilRisk': lambda x: not x == 'yes',
            #    'exposure': lambda x: x in ['single', 'multi'], # ignore
            'apyMean30d': lambda x: x > self.apy_floor
        }
        return pools[pools.apply(lambda x: all(v(x[k]) for k, v in pool_filters.items()), axis=1)]

class DynLst(FilteredDefiLlama):
    '''DynLst is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    reference_asset: str = 'eth'
    def __init__(self, config: dict = None):
        super().__init__(config)
        # get underlying history for performance
        shortlisted_tokens_ids = {
            '0xae7ab96520de3a18e5e111b5eaab095312d7fe84': '747c1d2a-c668-4682-b9f9-296708a3dd90',
            '0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0': '747c1d2a-c668-4682-b9f9-296708a3dd90',
            '0xae78736cd615f374d3085123a210448e74fc6393': 'd4b3c522-6127-4b89-bedf-83641cdcd2eb',
            '0x20bc832ca081b91433ff6c17f85701b6e92486c5': '66958f46-1d06-4f83-9fab-bbec354049d8',
            '0xac3e018457b222d93114458476f3e3416abbe38f': '77020688-e1f9-443c-9388-e51ace15cc32',
            '0xbe9895146f7af43049ca1c1ae358b0541ea49704': '0f45d730-b279-4629-8e11-ccb5cc3038b4',
            '0xf951e335afb289353dc249e82926178eac7ded78': 'ca2acc2d-6246-44aa-ae91-8725b2c62c7c',
            '0x856c4efb76c1d1ae02e20ceb03a2a6a08b0b8dc3': '423681e3-4787-40ce-ae43-e9f67c5269b3'}
        coros = [async_wrap(self.get_pool_hist_apy)(v)
                 for v in shortlisted_tokens_ids.values()]
        results = asyncio.run(safe_gather(coros))
        self.shortlisted_tokens_history = {k: results[i]['apy']/100
                                           for i, k in enumerate(shortlisted_tokens_ids)}
    def filter_underlyings(self):
        '''lowercase ! filter tokens that are not in the shortlist'''
        result = {'stETH': '0xae7ab96520de3a18e5e111b5eaab095312d7fe84',
                  'wstETH': '0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0',
                  'rETH': '0xae78736cd615f374d3085123a210448e74fc6393',
                  'rETH2': '0x20bc832ca081b91433ff6c17f85701b6e92486c5',
                  'frxETH': '0x5e8422345238f34275888049021821e8e08caa1f',
                  'sfrxETH': '0xac3e018457b222d93114458476f3e3416abbe38f',
                  'swETH': '0xf951e335afb289353dc249e82926178eac7ded78',
                  'ETH': '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
                  'null': '0x0000000000000000000000000000000000000000',
                  'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                  'cbETH': '0xbe9895146f7af43049ca1c1ae358b0541ea49704',
                  'oETH': '0x856c4efb76c1d1ae02e20ceb03a2a6a08b0b8dc3'}

        return {key: value.lower() for key, value in result.items()}
    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in ['Ethereum'],
            'project': lambda x: x in self.protocols,
            'underlyingTokens': lambda x: all(
                token.lower() in self.shortlisted_tokens.values() for token in x) if isinstance(x,
                                                                                                list) else False,
            'tvlUsd': lambda x: x > 1e5,
            #    'ilRisk': lambda x: not x == 'yes',
            #    'exposure': lambda x: x in ['single', 'multi'], # ignore
            #    'apyMean30d': lambda x: x>4
        }
        return pools[pools.apply(lambda x: all(v(x[k]) for k, v in pool_filters.items()), axis=1)]

    def underlying_apy(self,
                       underlyingTokens: [],
                       shortlisted_tokens_history: dict[str, pd.Series],
                       reserve_history: pd.DataFrame = None) -> pd.Series:
        apys: list[pd.Series] = []
        if reserve_history is None:
            # TODO: actually get reserve history....
            reserve_history = 1 / len(underlyingTokens)
        for u in underlyingTokens:
            u = u.lower()
            if u in shortlisted_tokens_history.keys():
                if isinstance(reserve_history, pd.DataFrame):
                    reserves = reserve_history[u]
                elif isinstance(reserve_history, numbers.Number):
                    reserves = reserve_history
                else:
                    raise TypeError

                apys.append(reserves * shortlisted_tokens_history[u])
        if len(apys) > 0:
            return pd.concat(apys, axis=1).sum(axis=1)
        else:
            # some random time series = 0
            return pd.Series(shortlisted_tokens_history['0xae7ab96520de3a18e5e111b5eaab095312d7fe84'] * 0.0)

    @ignore_error
    async def apy_history(self, metadata: dict, **kwargs) -> dict[str, pd.DataFrame]:
        '''gets various components of apy history from defillama'''
        dont_write_kwargs = deepcopy(kwargs)
        dont_write_kwargs.__delitem__('dirname')
        result = await super().apy_history(metadata, **kwargs)

        # From Defillama GUI it seems curve and convex already have underlying apy included.
        if not metadata['project'] in ['curve-finance', 'convex-finance']:
            # TODO: reserve ratio
            apy_underlyings = self.underlying_apy(metadata['underlyingTokens'], self.shortlisted_tokens_history)
            result['underlying_apy'] = apy_underlyings
            result['haircut_apy'] += apy_underlyings

        if 'dirname' in kwargs:
            await self.write_history(kwargs, metadata, result)

        return result
class DynYieldE(FilteredDefiLlama):
    '''DynLst is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    reference_asset: str = 'usd'
    def filter_underlyings(self) -> dict:
        '''lower case'''
        result = {'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
                  'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
                  'DAI': '0x6b175474e89094c44da98b954eedeac495271d0f',
                  'FRAX': '0x853d955acef822db058eb8505911ed77f175b99e',
                  'LUSD': '0x5f98805a4e8be255a32880fdec7f6728c6568ba0',
                  'MIM': '0x99d8a9c45b2eca8864373a26d1459e3dff1e17f3',
                  'aUSDC': '0xbcca60bb61934080951369a648fb03df4f96263c',
                  'aUSDT': '0x3ed3b47dd13ec9a98b44e6204a523e766b225811',
                  'aDAI': '0x028171bca77440897b824ca71d1c56cac55b68a3',
                  'cUSDC': '0x39aa39c021dfbae8fac545936693ac917d5e7563',
                  'cUSDT': '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9',
                  'cDAI': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643',
                  'sDAI': '0x83f20f44975d03b1b09e64809b757c47f942beea',
                  'eUSD': '0xdf3ac4f479375802a821f7b7b46cd7eb5e4262cc',
                  'crvUSD': '0xf939e0a03fb07f59a73314e73794be0e57ac1b4e',
                  'sUSD': '0x57ab1ec28d129707052df4df418d58a2d46d5f51',
                  'FRAXBP': '0x3175df0976dfa876431c2e9ee6bc45b65d3473cc',
                  'mkUSD': '0x4591dbff62656e7859afe5e45f6f47d3669fbb28'}
        return {key: value.lower() for key, value in result.items()}


    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in ['Ethereum'],
            'project': lambda x: x in self.protocols,
            'underlyingTokens': lambda x: all(
                token.lower() in self.shortlisted_tokens.values() for token in x) if isinstance(x,
                                                                                           list) else False,
            'tvlUsd': lambda x: x > 1e5,
            #    'ilRisk': lambda x: not x == 'yes',
            #    'exposure': lambda x: x in ['single', 'multi'], # ignore
            #    'apyMean30d': lambda x: x>4
        }
        return pools[pools.apply(lambda x: all(v(x[k]) for k, v in pool_filters.items()), axis=1)]


class DynYieldA(FilteredDefiLlama):
    '''DynLst is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    reference_asset: str = 'usd'
    def filter_underlyings(self) -> dict:
        '''lower case'''
        result = {'DAI': '0xda10009cbd5d07dd0cecc66161fc93d7c9000da1',
                  'FRAX': '0x17fc002b466eec40dae837fc4be5c67993ddbd6f',
                  'LUSD': '0x93b346b6bc2548da6a1e7d98e9a421b42541425b',
                  'MIM': '0xfea7a6a0b346362bf88a9e4a88416b77a57d6c2a',
                  'sUSD': '0xa970af1a584579b618be4d69ad6f73459d112f95',
                  'USDT': '0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9',
                  'USDC': '0xaf88d065e77c8cc2239327c5edb3a432268e5831'}
        return {key: value.lower() for key, value in result.items()}

    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in ['Arbitrum'],
            'project': lambda x: x in self.protocols,
            'underlyingTokens': lambda x: all(
                token.lower() in self.shortlisted_tokens.values() for token in x) if isinstance(x,
                                                                                           list) else False,
            'tvlUsd': lambda x: x > 1e5,
            #    'ilRisk': lambda x: not x == 'yes',
            #    'exposure': lambda x: x in ['single', 'multi'], # ignore
            #    'apyMean30d': lambda x: x>4
        }
        return pools[pools.apply(lambda x: all(v(x[k]) for k, v in pool_filters.items()), axis=1)]


class DynYieldB(FilteredDefiLlama):
    '''DynLst is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    reference_asset: str = 'usd'
    # shortlisted_tokens_tier2 = {
    #     'MIM': '0x99d8a9c45b2eca8864373a26d1459e3dff1e17f3',
    #     'GHO': '0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f',
    #     'sUSD': '0x57ab1ec28d129707052df4df418d58a2d46d5f51',
    #     'eUSD': '0xa0d69e286b938e21cbf7e51d71f6a4c8918f482f',
    #     'crvUSD': '0x8092ac8f4fe9e147098632482598f5855b25ee2f',
    # }

    def filter_underlyings(self) -> dict:
        result = {'USDC': '0x8ac76a51cc950d9822d68b83fe1ad97b32cd580d',
                'USDT': '0x55d398326f99059ff775485246999027b3197955',
                'DAI': '0x1af3f329e8be154074d8769d1ffa4ee058b1dbc3'}

        return {key: value.lower() for key, value in result.items()}

    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in ['BSC'],
            'project': lambda x: x in self.protocols,
            'underlyingTokens': lambda x: all(
                token.lower() in self.shortlisted_tokens.values() for token in x) if isinstance(x,
                                                                                           list) else False,
            'tvlUsd': lambda x: x > 1e5,
            #    'ilRisk': lambda x: not x == 'yes',
            #    'exposure': lambda x: x in ['single', 'multi'], # ignore
            #    'apyMean30d': lambda x: x>4
        }
        return pools[pools.apply(lambda x: all(v(x[k]) for k, v in pool_filters.items()), axis=1)]


class DynYieldBTCE(FilteredDefiLlama):
    '''DynLst is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    reference_asset: str = 'btc'

    def filter_underlyings(self) -> dict:
        result = {'WBTC': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599'}
        return {key: value.lower() for key, value in result.items()}

    def filter_pools(self, pools: pd.DataFrame) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in ['Ethereum'],
            'project': lambda x: x in self.protocols,
            'underlyingTokens': lambda x: all(
                token.lower() in self.shortlisted_tokens.values() for token in x) if isinstance(x,
                                                                                           list) else False,
            'tvlUsd': lambda x: x > 3e6,
            #    'ilRisk': lambda x: not x == 'yes',
            #    'exposure': lambda x: x in ['single', 'multi'], # ignore
            #    'apyMean30d': lambda x: x>4
        }
        return pools[pools.apply(lambda x: all(v(x[k]) for k, v in pool_filters.items()), axis=1)]


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
        # Create a DefiLlama instance
        if sys.argv[2] in ['snapshot', 'continuous']:
            pools = pd.concat([DynLst().pools,
                               DynYieldE().pools,DynYieldB().pools], axis=0)['pool'].tolist()
            if sys.argv[2] == 'snapshot':
                filename = os.path.join(os.sep, os.getcwd(), 'data', 'snapshots',
                                        'defillama_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
                defillama = DefiLlamaLiveScrapper(filename=filename,
                                                  pool_filter=(lambda x: x['pool'] in pools))
                asyncio.run(defillama.snapshot())
                quit()
            elif sys.argv[2] == 'continuous':
                filename = os.path.join(os.sep, os.getcwd(), 'data',
                                        'defillama_live.csv')
                defillama = DefiLlamaLiveScrapper(filename=filename,
                                                  pool_filter=(lambda x: x['pool'] in pools))
                asyncio.run(defillama.start(frequency=timedelta(minutes=5)))
            else:
                raise NotImplementedError

        elif sys.argv[2] == 'DiscoveryDefiLlama':
            if len(sys.argv) >= 6:
                kwargs = {'apy_floor': float(sys.argv[3]),
                          'tvlUsd_floor': float(sys.argv[4]),
                          'chains': sys.argv[5:]}
            else:
                kwargs = {}
            defillama = DiscoveryDefiLlama(**kwargs)
        else:
            if sys.argv[2] in globals():
                with open(sys.argv[3], 'r') as fp:
                    config = yaml.safe_load(fp)
                defillama = globals()[sys.argv[2]](config)
            else:
                raise ValueError


        # prepare history arguments
        dirname = os.path.join(os.sep, os.getcwd(), 'data', sys.argv[2])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        date_format = '%Y-%m-%d %H:%M:%S %Z'
        end = datetime.now().replace(tzinfo=timezone.utc)
        start = end - timedelta(days=365)
        history_kwargs = {'dirname': dirname,
                          #'reward_history': None
                              # {'discount_lookback': timedelta(days=90),
                              #  'end': end.strftime(date_format),
                              #  'end_format': date_format,
                              #  'span': 900,
                              #  'period': '1d'}
                          }

        # fetch everything asynchronously
        keys = defillama.pools['pool'].tolist()
        coros = [defillama.apy_history(x, **history_kwargs) for _, x in defillama.pools.iterrows()]
        all_history = defillama.all_apy_history(**history_kwargs)

        if False:
            apy = compute_moments(all_history)
            top_pools = print_top_pools(apy)
            historical_best_pools = get_historical_best_pools(apy, 10, start, end).resample('d').apply('mean')
            historical_best_pools.index = [t.replace(tzinfo=None) for t in historical_best_pools.index]
            mean_best_history = historical_best_pools.mean(axis=1)
            ever_been_top = defillama.pools[defillama.pools['pool'].isin(historical_best_pools.columns)]

            filename = os.path.join(os.sep, os.getcwd(), f'best_ever_{sys.argv[2]}.xlsx')
            with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
                mean_best_history.to_excel(writer, 'mean_best_history')
                ever_been_top.to_excel(writer, 'ever_been_top')