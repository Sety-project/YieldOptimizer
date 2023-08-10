#!/usr/bin/env python
import logging
import numbers
import os
import sys
import asyncio
from copy import deepcopy
from utils.async_utils import async_wrap, safe_gather
from utils.io_utils import ignore_error
from datetime import datetime, timedelta, timezone
from abc import abstractmethod
import pandas as pd
from defillama2 import DefiLlama


class FilteredDefiLlama(DefiLlama):
    '''
    filters protocols and pools from defillama
    '''
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger('defillama_history')
        super().__init__(*args, **kwargs)
        protocols = self.get_protocols()
        protocols['name'] = protocols['name'].apply(lambda s: s.lower().replace(' ', '-'))
        self.protocols = self.filter_protocols(protocols)
        self.shortlisted_tokens = self.filter_underlyings()
        self.pools = self.filter_pools(self.get_pools_yields())

    @abstractmethod
    def filter_underlyings(self) -> dict:
        '''filter underlyings'''
        raise NotImplementedError

    def filter_protocols(self, protocols, **kwargs) -> pd.DataFrame:
        '''filter protocols'''
        excluded_categories = ['NFT Lending', 'NFT Marketplace']
        excluded_protocols = ['uwu-lend', 'sturdy', 'goldfinch', 'ribbon', 'idle', 'ipor', 'notional', 'merkl', 'asymetrix-protocol']
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

    @abstractmethod
    def filter_pools(self, pools, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @ignore_error
    async def apy_history(self, metadata: dict,  **kwargs) -> dict[str, pd.DataFrame]:
        '''gets various components of apy history from defillama'''

        # get pool history
        pool_history = await async_wrap(defillama.get_pool_hist_apy)(metadata['pool'])
        apy = pool_history['apy']

        apyReward = pool_history['apyReward']

        reward_discount = 0
        if kwargs is not None and 'reward_history' in kwargs:
            if metadata['rewardTokens'] not in [None, []]:
                token_addrs_n_chains = {rewardToken: metadata['chain'] for rewardToken in metadata['rewardTokens']}
                reward_discount = await self.discount_reward_by_minmax(token_addrs_n_chains, **kwargs)

        # pathetic attempt at IL...
        il = pool_history['il7d'].fillna(0) * 52
        tvl = pool_history['tvlUsd']

        haircut_apy = apy - reward_discount * apyReward.fillna(0) - il.fillna(0)

        result = pd.DataFrame({'haircut_apy': haircut_apy,
                'apy': apy,
                'apyReward': apyReward,
                'il': il,
                'tvl': tvl})

        if 'dirname' in kwargs:
            name = os.path.join(kwargs['dirname'], '{}.csv'.format(metadata['pool']))
            result.to_csv(name, mode='w', header=not os.path.isfile(name))

        return result

    @ignore_error
    async def discount_reward_by_minmax(self, token_addrs_n_chains, **kwargs) -> pd.DataFrame:
        kwargs_reward_history = deepcopy(kwargs['reward_history'])
        discount_lookback = kwargs_reward_history.pop('discount_lookback')

        reward_history = await async_wrap(self.get_prices_at_regular_intervals)(token_addrs_n_chains,
                                                                                **kwargs_reward_history)

        reward_discount = pd.DataFrame()
        for rewardToken in reward_history.columns:
            # need to clean defillama data
            token_history = reward_history[rewardToken].dropna()
            # discount by 30d max-min
            reward_discount[rewardToken] = 1 - token_history.rolling(discount_lookback).apply(
                lambda x: (max(x) - min(x))) / token_history
        # defillama doesnt breakdown rewards so take the min...
        reward_discount = reward_discount.min(axis=1)
        return reward_discount

    def find_underlyings(self, pools: pd.DataFrame) -> pd.DataFrame:
        '''find underlying tokens of pools'''
        pools['newUnderlyingTokens'] = pools['underlyingTokens'].apply(
            lambda x: [token for token in x if token.lower() not in self.shortlisted_tokens.values()]
            if x is not None else None)
        return pools

    def all_apy_history(self, **kwargs) -> pd.DataFrame:
        metadata = [x.to_dict() for _, x in self.pools.iterrows()]
        coros = [self.apy_history(meta, **kwargs) for meta in metadata]
        data = asyncio.run(safe_gather(coros))

        if 'dirname' in kwargs:
            filename = os.path.join(kwargs['dirname'], 'pool_metadata.csv')
            pd.DataFrame(metadata).set_index('pool').to_csv(filename, mode='w')

        return pd.DataFrame({key['pool']: value['haircut_apy'] for key, value in zip(metadata, data)})


class DynLiq(FilteredDefiLlama):
    '''DynLiq is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get underlying history for performance
        shortlisted_tokens_ids = {
            '0xae7ab96520de3a18e5e111b5eaab095312d7fe84': '747c1d2a-c668-4682-b9f9-296708a3dd90',
            '0xae78736cd615f374d3085123a210448e74fc6393': 'd4b3c522-6127-4b89-bedf-83641cdcd2eb',
            '0x20bc832ca081b91433ff6c17f85701b6e92486c5': '66958f46-1d06-4f83-9fab-bbec354049d8',
            '0xac3e018457b222d93114458476f3e3416abbe38f': '77020688-e1f9-443c-9388-e51ace15cc32',
            '0xbe9895146f7af43049ca1c1ae358b0541ea49704': '0f45d730-b279-4629-8e11-ccb5cc3038b4'}
        self.shortlisted_tokens_history = {k: self.get_pool_hist_apy(v)['apy'] for k, v in
                                           shortlisted_tokens_ids.items()}

    def filter_underlyings(self):
        '''filter tokens that are not in the shortlist'''
        return {'stETH': '0xae7ab96520de3a18e5e111b5eaab095312d7fe84',
         'wstETH': '0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0',
         'rETH': '0xae78736cd615f374d3085123a210448e74fc6393',
         'rETH2': '0x20bc832ca081b91433ff6c17f85701b6e92486c5',
         'frxETH': '0x5e8422345238f34275888049021821e8e08caa1f',
         'sfrxETH': '0xac3e018457b222d93114458476f3e3416abbe38f',
         'ETH': '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE'.lower(),
         'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'}
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

    async def apy_history(self, metadata: dict, **kwargs) -> dict[str, pd.DataFrame]:
        dont_write_kwargs = deepcopy(kwargs)
        dont_write_kwargs.__delitem__('dirname')
        result = await super().apy_history(metadata, **dont_write_kwargs)
        # add underlying apy
        # TODO: reserve ratio
        apy_underlyings = self.underlying_apy(metadata['underlyingTokens'], self.shortlisted_tokens_history) if hasattr(
            self, 'underlying_apy') else None

        result['underlying_apy'] = apy_underlyings
        # TODO: it depends :(
        #  result['haircut_apy'] += apy_underlyings

        if 'dirname' in kwargs:
            name = os.path.join(kwargs['dirname'], '{}.csv'.format(metadata['pool']))
            result.to_csv(name, mode='w', header=not os.path.isfile(name))

        return result
class DynYieldE(FilteredDefiLlama):
    '''DynLiq is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    # shortlisted_tokens_tier2 = {
    #     'MIM': '0x99d8a9c45b2eca8864373a26d1459e3dff1e17f3',
    #     'GHO': '0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f',
    #     'sUSD': '0x57ab1ec28d129707052df4df418d58a2d46d5f51',
    #     'eUSD': '0xa0d69e286b938e21cbf7e51d71f6a4c8918f482f',
    #     'crvUSD': '0x8092ac8f4fe9e147098632482598f5855b25ee2f',
    # }

    def filter_underlyings(self) -> dict:
        return {'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
                                   'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
                                   'DAI': '0x6b175474e89094c44da98b954eedeac495271d0f',
                                   'BUSD': '0x4fabb145d64652a948d72533023f6e7a623c7c53',
                                   'FRAX': '0x853d955acef822db058eb8505911ed77f175b99e',
                                   'LUSD': '0x5f98805a4e8be255a32880fdec7f6728c6568ba0',
                                   'FRAXBP': '0x3175df0976dfa876431c2e9ee6bc45b65d3473cc',
                                   '3CRV': '0x6c3f90f043a72fa612cbac8115ee7e52bde6e490',
                                   'sDAI': '0x83f20f44975d03b1b09e64809b757c47f942beea',
                                   'aUSDC': '0xbcca60bb61934080951369a648fb03df4f96263c',
                                   'aUSDT': '0x3ed3b47dd13ec9a98b44e6204a523e766b225811',
                                   'aDAI': '0x028171bca77440897b824ca71d1c56cac55b68a3',
                                   'cUSDC': '0x39aa39c021dfbae8fac545936693ac917d5e7563',
                                   'cUSDT': '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9',
                                   'cDAI': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643'}

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


class DynYieldB(FilteredDefiLlama):
    '''DynLiq is a class that filters pools and
    stores the historical apy of the pool and the historical apy of the underlying tokens
    '''
    # shortlisted_tokens_tier2 = {
    #     'MIM': '0x99d8a9c45b2eca8864373a26d1459e3dff1e17f3',
    #     'GHO': '0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f',
    #     'sUSD': '0x57ab1ec28d129707052df4df418d58a2d46d5f51',
    #     'eUSD': '0xa0d69e286b938e21cbf7e51d71f6a4c8918f482f',
    #     'crvUSD': '0x8092ac8f4fe9e147098632482598f5855b25ee2f',
    # }

    def filter_underlyings(self) -> dict:
        return {'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
                                   'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
                                   'DAI': '0x6b175474e89094c44da98b954eedeac495271d0f',
                                   'BUSD': '0x4fabb145d64652a948d72533023f6e7a623c7c53',
                                   'FRAX': '0x853d955acef822db058eb8505911ed77f175b99e',
                                   'LUSD': '0x5f98805a4e8be255a32880fdec7f6728c6568ba0',
                                   'FRAXBP': '0x3175df0976dfa876431c2e9ee6bc45b65d3473cc',
                                   '3CRV': '0x6c3f90f043a72fa612cbac8115ee7e52bde6e490',
                                   'sDAI': '0x83f20f44975d03b1b09e64809b757c47f942beea',
                                   'aUSDC': '0xbcca60bb61934080951369a648fb03df4f96263c',
                                   'aUSDT': '0x3ed3b47dd13ec9a98b44e6204a523e766b225811',
                                   'aDAI': '0x028171bca77440897b824ca71d1c56cac55b68a3',
                                   'cUSDC': '0x39aa39c021dfbae8fac545936693ac917d5e7563',
                                   'cUSDT': '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9',
                                   'cDAI': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643'}

    def filter_pools(self, pools) -> pd.DataFrame:
        # shortlist pools
        pool_filters = {
            'chain': lambda x: x in ['BSC'],
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
        elif sys.argv[2] == 'DynYieldB':
            defillama = DynYieldB()

        # prepare history arguments
        dirname = os.path.join(os.getcwd(), 'data', sys.argv[2])
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        date_format = '%Y-%m-%d %H:%M:%S %Z'
        history_kwargs = {'dirname': dirname,
                          'reward_history':
                              {'discount_lookback': timedelta(days=30),
                               'end': datetime(2023, 7, 30, tzinfo=timezone.utc).strftime(date_format),
                               'end_format': date_format,
                               'span': 180,
                               'period': '1d'}
                          }

        # fetch everything asynchronously
        keys = defillama.pools['pool'].tolist()
        coros = [defillama.apy_history(x, **history_kwargs) for _, x in defillama.pools.iterrows()]
        apy = defillama.all_apy_history(**history_kwargs)
        # apply apy cutoff date and cap 200%
        apy = apy[apy.index < datetime.now().replace(tzinfo=timezone.utc)]
        apy = apy.applymap(lambda x: min(x, 200))

        # compute moments
        ewm_df = apy.ewm(halflife=timedelta(days=7), times=apy.index)

        # compute top pools
        top_pools = defillama.pools[['project', 'symbol', 'pool', 'tvlUsd']].set_index('pool').join(ewm_df.mean().iloc[-1])
        top_pools.rename(columns={top_pools.columns[-1]: '7d avg discounted apy'}, inplace=True)
        top_pools = top_pools[top_pools['7d avg discounted apy']].sort_values('7d avg discounted apy', ascending=False)

        # save to excel
        try:
            with pd.ExcelWriter('defillama_hist.xlsx', engine='openpyxl', mode='a') as writer:
                top_pools.to_excel(writer, datetime.now().strftime(f"{sys.argv[2]} %d %m %Y %H_%M_%S"))
        except PermissionError:
            print('Please close the excel file')
            exit(0)