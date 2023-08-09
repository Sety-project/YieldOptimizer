#!/usr/bin/env python
import functools
import importlib
import numbers
import os
import sys
import pickle
import asyncio
from utils.async_utils import async_wrap, safe_gather
from utils.io_utils import async_to_csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from defillama2 import DefiLlama


class FilteredDefiLlama(DefiLlama):
    '''
    filters protocols and pools from defillama
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        protocols = self.get_protocols()
        protocols['name'] = protocols['name'].apply(lambda s: s.lower().replace(' ', '-'))
        self.protocols = self.filter_protocols(protocols)
        self.pools = self.filter_pools(self.get_pools_yields())

    @abstractmethod
    def filter_protocols(self, protocols, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def filter_pools(self, pools, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    async def apy_history(self, metadata: dict,  **kwargs) -> dict[str, pd.DataFrame]:
        '''gets various components of apy history from defillama'''
        # get pool history
        pool_history = await async_wrap(defillama.get_pool_hist_apy)(metadata['pool'])
        apy = pool_history['apy']

        # TODO: reserve ratio
        apy_underlyings = self.underlying_apy(metadata['underlyingTokens'], self.shortlisted_tokens_history) if hasattr(self, 'underlying_apy') else None
        apyReward = pool_history['apyReward']

        # TODO: haircut rewards for hedging or conservative mark
        '''need an oracle for rewards...
        coingecko = CoinGeckoClient()
        reward_coins = ['curve-dao-token', 'stake-dao', 'pendle', 'convex-finance', 'unsheth']
        reward_history = pd.concat({coin: sum(pd.DataFrame(self.get_tokens_hist_prices(
        {reward_coin:reward_coin, reward_coin:'coingecko'}).set_index('t') 
                   for coin in reward_coins)}, axis=1)
        '''
        if kwargs is not None and 'reward_history' in kwargs:
            reward_dict = dict()
            if metadata['rewardTokens'] not in [None, []]:
                token_addrs_n_chains = {rewardToken: metadata['chain'] for rewardToken in metadata['rewardTokens']}
                reward_dict[metadata['pool']] = self.get_prices_at_regular_intervals(token_addrs_n_chains,
                                                                              **kwargs['reward_history'])
        # pools['rewardValo'] = pools['rewardTokens']
        # pathetic attempt at IL...
        il = pool_history['il7d'].fillna(0) * 52
        tvl = pool_history['tvlUsd']

        haircut_apy = apy \
                      + (apy_underlyings.fillna(0) if apy_underlyings is not None else 0) \
                      - 0.3 * apyReward.fillna(0) - 0 * il.fillna(0)

        result = pd.DataFrame({'haircut_apy': haircut_apy,
                'apy': apy,
                'apy_underlyings': apy_underlyings,
                'apyReward': apyReward,
                'il': il,
                'tvl': tvl})

        if kwargs['dirname'] is not None:
            name = os.path.join(kwargs['dirname'], '{}.csv'.format(metadata['pool']))
            result.to_csv(name, mode='a', header=not os.path.isfile(name))

        return result

    def all_apy_history(self, **kwargs) -> pd.DataFrame:
        metadata = [x.to_dict() for _, x in self.pools.iterrows()]
        coros = [self.apy_history(meta, **kwargs) for meta in metadata]
        data = asyncio.run(safe_gather(coros))

        if kwargs['dirname'] is not None:
            filename = os.path.join(kwargs['dirname'], 'pool_metadata.csv')
            pd.DataFrame(metadata).set_index('pool').to_csv(filename, mode='w')

        return pd.DataFrame({key['pool']: value['haircut_apy'] for key, value in zip(metadata, data)})


class DynLiq(FilteredDefiLlama):
    '''DynLiq is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    def __init__(self, *args, **kwargs):
        # filter tokens (hard coded)
        self.shortlisted_tokens = {'stETH': '0xae7ab96520de3a18e5e111b5eaab095312d7fe84',
                                  'wstETH': '0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0',
                                  'rETH': '0xae78736cd615f374d3085123a210448e74fc6393',
                                  'rETH2': '0x20bc832ca081b91433ff6c17f85701b6e92486c5',
                                  'frxETH': '0x5e8422345238f34275888049021821e8e08caa1f',
                                  'sfrxETH': '0xac3e018457b222d93114458476f3e3416abbe38f',
                                  'ETH': '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE'.lower(),
                                  'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'}
        #                      'cbETH': '0xbe9895146f7af43049ca1c1ae358b0541ea49704'}
        shortlisted_tokens_ids = {
            '0xae7ab96520de3a18e5e111b5eaab095312d7fe84': '747c1d2a-c668-4682-b9f9-296708a3dd90',
            '0xae78736cd615f374d3085123a210448e74fc6393': 'd4b3c522-6127-4b89-bedf-83641cdcd2eb',
            '0x20bc832ca081b91433ff6c17f85701b6e92486c5': '66958f46-1d06-4f83-9fab-bbec354049d8',
            '0xac3e018457b222d93114458476f3e3416abbe38f': '77020688-e1f9-443c-9388-e51ace15cc32',
            '0xbe9895146f7af43049ca1c1ae358b0541ea49704': '0f45d730-b279-4629-8e11-ccb5cc3038b4'}
        super().__init__(*args, **kwargs)
        self.shortlisted_tokens_history = {k: self.get_pool_hist_apy(v)['apy'] for k, v in
                                           shortlisted_tokens_ids.items()}

    def filter_protocols(self, protocols, **kwargs) -> pd.DataFrame:
        '''filter protocols'''
        excluded_categories = ['NFT Lending', 'NFT Marketplace', 'Bridge']
        excluded_protocols = ['ribbon', 'merkl', 'asymetrix-protocol', 'across']
        protocol_filters = {
            #    'chain': lambda x: x in ['Ethereum','Multi-Chain'],
            'name': lambda x: x not in excluded_protocols,
            #    'audits': lambda x: x in [None, '0', '1', '2', '3'],
            'category': lambda x: x not in excluded_categories,
            #    'listedAt': lambda x: not x>datetime(2023,3,1).timestamp()*1000, # 1mar23
            #    'tvl': lambda x: x>10e6,
            'openSource': lambda x: not x == False,
            #    'wrongLiquidity': lambda x: x==True,
            #    'rugged': lambda x: x==True,
        }
        return protocols[protocols.apply(lambda x: all(v(x[k]) for k, v in protocol_filters.items()), axis=1)]

    def filter_pools(self, pools) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in ['Ethereum'],
            'project': lambda x: x in self.protocols['name'].unique(),
            'underlyingTokens': lambda x: all(
                token.lower() in self.shortlisted_tokens.values() for token in x) if isinstance(x,
                                                                                           list) else False,
            'tvlUsd': lambda x: x > 3e6,
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

        return result

'''
main code
'''

if __name__ == '__main__':
    if sys.argv[1] == 'defillama':
        # Create a DefiLlama instance
        if sys.argv[2] == 'DynLiq':
            defillama = DynLiq()
        elif sys.argv[2] == 'DynYieldE':
            defillama = DynYieldE()

        # date_format = '%Y-%m-%d %H:%M:%S %Z'
        # we wish we could get reward history from defillama, but their session doesn't work??...
        # and anyway defillama doesn't split rewards by rewardToken...
        # history_kwargs = {'reward_history':
        #                       {'end': datetime(2023, 7, 30, tzinfo=timezone.utc).strftime(date_format),
        #                        'end_format': date_format,
        #                        'span': 180,
        #                        'period': '1d'}
        #                   }

        dirname = os.path.join(os.getcwd(), 'data', 'dynliq')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        history_kwargs = {'dirname': dirname}

        keys = defillama.pools['pool'].tolist()
        coros = [defillama.apy_history(x, **history_kwargs) for _, x in defillama.pools.iterrows()]
        apy = defillama.all_apy_history(**history_kwargs)
        # apply apy cutoff date and cap 200%
        apy = apy[apy.index < datetime.now().replace(tzinfo=timezone.utc)]
        apy = apy.applymap(lambda x: min(x, 200))
        # compute moments
        ewm_df = apy.ewm(halflife=timedelta(days=7), times=apy.index)

        top_pools = defillama.pools[['project', 'symbol', 'pool', 'tvlUsd']].set_index('pool').join(ewm_df.mean().iloc[-1])
        top_pools.rename(columns={top_pools.columns[-1]: '7d avg discounted apy'}, inplace=True)
        top_pools = top_pools[top_pools['7d avg discounted apy'] > 5].sort_values('7d avg discounted apy', ascending=False)

        try:
            with pd.ExcelWriter('defillama_hist.xlsx', engine='openpyxl', mode='a') as writer:
                top_pools.to_excel(writer, datetime.now().strftime("%d %m %Y %H_%M_%S"))
        except PermissionError:
            print('Please close the excel file')
            exit(1)