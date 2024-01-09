import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import wraps, lru_cache

import pandas as pd
import pycoingecko


#from cache_to_disk import cache_to_disk

def cache_to_file(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_file = os.path.join(os.path.dirname(__file__), func.__name__ + ''.join(args[1:]) + ''.join(str(kwargs)))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                try:
                    cached_data = pickle.load(f)
                except Exception as e:
                    print(f"Error loading cache file: {e}")
                    cached_data = None

            if cached_data is not None and cached_data['args'] == args and cached_data['kwargs'] == kwargs:
                return cached_data['result']

        result = func(*args, **kwargs)

        with open(cache_file, 'wb') as f:
            pickle.dump({'args': args, 'kwargs': kwargs, 'result': result}, f)

        return result

    return wrapper


def apply_decorator(cls):
    for name, method in pycoingecko.CoinGeckoAPI.__dict__.items():
        if callable(method) and 'fetch_' in name:
            setattr(cls, name, lru_cache(method))
    return cls
#
# logger = logging.getLogger('cache_to_disk')
# level = logging.DEBUG
# handler = logging.StreamHandler()
# handler.setLevel(level)
# handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
# logger.addHandler(handler)

@apply_decorator
class myCoinGeckoAPI(pycoingecko.CoinGeckoAPI):
    defillama_mapping = ({'id': 'id',
                         'symbol': 'symbol',
                         'name': 'name'}|
                         {'alephium': 'Alephium',
                          'algorand': 'Algorand',
                          'aptos': 'Aptos',
                          'arbitrum-nova': 'Arbitrum Nova',
                          'arbitrum-one': 'Arbitrum',
                          'astar': 'Astar',
                          'aurora': 'Aurora',
                          'avalanche': 'Avalanche',
                          'base': 'Base',
                          'binance-smart-chain': 'Binance',
                          'bitcoin-cash': 'Bitcoin',
                          'bitgert': 'Bitgert',
                          'bittorrent': 'Bittorrent',
                          'boba': 'Boba',
                          'canto': 'Canto',
                          'cardano': 'Cardano',
                          'celo': 'Celo',
                          'conflux': 'Conflux',
                          'core': 'CORE',
                          'cosmos': 'Cosmos',
                          'cronos': 'Cronos',
                          'cube': 'Cube',
                          'dogechain': 'Dogechain',
                          'elastos': 'Elastos',
                          'elrond': 'Elrond',
                          'empire': 'Empire',
                          'eos': 'EOS',
                          'eos-evm': 'EOS EVM',
                          'ethereum': 'Ethereum',
                          'ethereumpow': 'EthereumPoW',
                          'everscale': 'Everscale',
                          'evmos': 'Evmos',
                          'fantom': 'Fantom',
                          'fuse': 'Fuse',
                          'fusion-network': 'Fusion',
                          'harmony-shard-0': 'Harmony',
                          'hedera-hashgraph': 'Hedera',
                          'hoo': 'Hoo',
                          'injective': 'Injective',
                          'iotex': 'IoTeX',
                          'kava': 'Kava',
                          'kucoin-community-chain': 'Kucoin',
                          'kujira': 'Kujira',
                          'kusama': 'Kusama',
                          'linea': 'Linea',
                          'manta-pacific': 'Manta',
                          'mantle': 'Mantle',
                          'meter': 'Meter',
                          'metis-andromeda': 'Metis',
                          'milkomeda-cardano': 'Milkomeda',
                          'moonbeam': 'Moonbeam',
                          'moonriver': 'Moonriver',
                          'near-protocol': 'Near',
                          'neo': 'NEO',
                          'nuls': 'Nuls',
                          'oasis': 'Oasis',
                          'ontology': 'Ontology',
                          'optimistic-ethereum': 'Optimism',
                          'osmosis': 'Osmosis',
                          'polkadot': 'Polkadot',
                          'polygon-pos': 'Polygon',
                          'polygon-zkevm': 'Polygon zkEVM',
                          'rollux': 'Rollux',
                          'ronin': 'Ronin',
                          'scroll': 'Scroll',
                          'secret': 'Secret',
                          'smartbch': 'smartBCH',
                          'solana': 'Solana',
                          'sui': 'Sui',
                          'syscoin': 'Syscoin',
                          'telos': 'Telos',
                          'terra': 'Terra',
                          'tezos': 'Tezos',
                          'theta': 'Theta',
                          'thorchain': 'Thorchain',
                          'thundercore': 'ThunderCore',
                          'tomochain': 'TomoChain',
                          'tron': 'Tron',
                          'velas': 'Velas',
                          'wanchain': 'Wanchain',
                          'waves': 'Waves',
                          'xdai': 'xDai',
                          'xdc-network': 'XDC',
                          'zilliqa': 'Zilliqa',
                          'zksync': 'zkSync Era'})

    def __init__(self):
        super().__init__()
        filename = os.path.join(os.sep, os.getcwd(), 'config', 'coingecko_address_map.csv')
        if not os.path.isfile(filename):
            ids = self.get_coins_list(include_platform='true')
            address_map = pd.concat([pd.Series(x['platforms']
                                               | {'id': x['id'],
                                                  'symbol': x['symbol'],
                                                  'name': x['name']})
                                     for x in ids], axis=1).T
            address_map.columns = address_map.columns
            self.address_map = address_map.fillna('').set_index('id')
            self.address_map.to_csv(filename)
        else:
            self.address_map = pd.read_csv(filename, index_col='id')

    def adapt_address_map_to_defillama(self) -> None:
        # keep only columns that are in the chain mapping
        table = self.address_map.filter(myCoinGeckoAPI.defillama_mapping.keys())
        # rename columns to defillama chain names
        table.columns = [myCoinGeckoAPI.defillama_mapping[chain] for chain in table.columns]
        self.address_map = table

    def address_to_id(self, address: str, chain: str) -> str:
        return self.address_map[self.address_map[chain] == address].index[0]

    def address_to_symbol(self, address: str, chain: str) -> str:
        temp = self.address_map.loc[self.address_map[chain] == address, 'symbol'].squeeze()
        return temp if type(temp) == str else ''

    def fetch_range(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        '''
        :param symbol: coin name
        :param start: start date
        :param end: end date
        :return: df with columns: timestamp, open, high, low, close, volume
        '''
        df = pd.DataFrame(self.get_coin_market_chart_range_by_id(id=symbol, vs_currency='usd',
                                                                 from_timestamp=start.timestamp(),
                                                                 to_timestamp=end.timestamp()),
                          columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x / 1000).replace(tzinfo=timezone.utc))
        return df

    def fetch_market_chart(self, _id: str, days: int, vs_currency='usd') -> pd.DataFrame:
        '''
        :param symbol: coin name
        :param start: start date
        :param end: end date
        :return: df with columns: timestamp, open, high, low, close, volume
        '''
        df = pd.DataFrame(self.get_coin_market_chart_by_id(id=_id, days=days, vs_currency=vs_currency)['prices'],
                          columns=['timestamp', 'price'])
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x / 1000).replace(tzinfo=timezone.utc))
        return df

    def fetch_ohlc(self, id_: str, days: int, vs_currency='usd') -> pd.DataFrame:
        '''
        :param symbol: coin name
        :param start: start date
        :param end: end date
        :return: df with columns: timestamp, open, high, low, close, volume
        '''
        if days <= 2:
            freq = timedelta(minutes=30)
        elif days <= 30:
            freq = timedelta(hours=4)
        else:
            freq = timedelta(days=4)

        df = pd.DataFrame(self.get_coin_ohlc_by_id(id=id_, vs_currency=vs_currency,
                                                   days=str(days)),
                          columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x / 1000).replace(tzinfo=timezone.utc))
        return df

if __name__ == '__main__':
    if sys.argv[1] =='ohlcv':
        end = datetime.now()
        start = end - pd.Timedelta(days=30)
        cg = myCoinGeckoAPI()
        df = cg.fetch_ohlc(sys.argv[2], vs_currency='usd', days=7)
        df.to_csv('_'.join(sys.argv[1:]) + '.csv')
    else:
        peers = ['celestia',
                 'tia',
                 'conflux-token',
                 'enjincoin',
                 'injective-protocol',
                 'zencash',
                 'stargate-finance',
                 'celo',
                 'celo-wormhole',
                 'moviebloc',
                 'axie-infinity',
                 'kava',
                 'blur',
                 'cyberconnect',
                 'cyberpunk-city',
                 'algorand']
        ecosystem = ['singularitynet', 'singularitydao', 'hypercycle', 'rejuve-ai', 'nunet', 'sophiaverse']

        cg = myCoinGeckoAPI()
        tickers = cg.get_coin_ticker_by_id('bitcoin')['tickers']
        df_list = []
        for ids in (peers + ecosystem):
            i = 0
            while i < 10:
                try:
                    ticker = cg.get_coin_ticker_by_id(ids)
                    price = cg.get_price(ids, 'usd', include_market_cap=True)
                    new_data = pd.DataFrame(ticker['tickers'])
                    new_data['price_usd'] = price[ids]['usd']
                    new_data['market_cap'] = price[ids]['usd_market_cap']
                    df_list.append(new_data)
                    print(f'{ids} done')
                    i=10
                except Exception as e:
                    print(str(e))
                    i = i+1
                    time.sleep(60)

        df = pd.concat(df_list)