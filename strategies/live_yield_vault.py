import asyncio
import logging

import pandas as pd
import requests

from research.research_engine import build_ResearchEngine
from strategies.vault_betsizing import YieldStrategy
from utils.api_utils import build_logging
from utils.async_utils import async_wrap, safe_gather

build_logging('defillama_live')

class LiveYieldVaultRebalancing:
    # TODO: fill in this dict to translate defillama pool_id to strategy_id
    strategy_ids = {}
    def __init__(self, vault_id: str, parameters: dict):
        self.vault_id = vault_id
        self.parameters = parameters
        self.live_feed = None  # TODO: initialize parameters['input_data']['live_feed']
    def run_live(self):
        # load data and fit model. quite wasteful since it recomputes ewa at all steps...
        engine = build_ResearchEngine(self.parameters)
        performance = pd.concat(engine.performance, axis=1)
        rebalancing_strategy = YieldStrategy(research_engine=engine, params=self.parameters['strategy'])

        '''
        read allocations from server
        define weights in the order of parameters['input_data']['selected_instruments']
        assign 0 if absent
        unwind any strategies in allocations but not in parameters['input_data']['selected_instruments'] 
        '''
        strategy_indices = [LiveYieldVaultRebalancing.strategy_ids[instrument] for instrument in
                            self.parameters['input_data']['selected_instruments']]
        allocations = requests.get(
            f'https://automated-execution-server-v2.dev.singularitydao.ai/execution-server/dynaset-vaults/allocation/{self.vault_id}').json()

        rebalancing_strategy.state.wealth = sum(x['usdValue'] for x in allocations)
        rebalancing_strategy.state.weights = []
        for strategy_index in strategy_indices:
            allocation = next((x['usdValue'] for x in allocations if x['id'] == strategy_index), 0)
            if allocation == 0:
                logging.getLogger('defillama_live').warning(f'new strategy {strategy_index}')
            rebalancing_strategy.state.weights.append(allocation)

        removed_strategies = [(x['id'], x['usdValue']) for x in allocations if x['id'] not in strategy_indices]
        if len(removed_strategies) > 0:
            logging.getLogger('defillama_live').warning(f'removed strategies: {removed_strategies}')

        # optimize on latest index
        predicted_apys = rebalancing_strategy.predict(performance.index[-1])
        optimal_weights = rebalancing_strategy.optimal_weights(predicted_apys)
        assert len(strategy_indices) == len(optimal_weights), 'Number of instruments in index does not match number of weights'

        # trade
        trades = [async_wrap(self.make_trade)(strategy_index, optimal_weights[i] - rebalancing_strategy.state.weights[i])
                  for i, strategy_index in enumerate(strategy_indices)]
        unwinds = [async_wrap(self.make_trade)(strategy_index, -usdValue) for strategy_index, usdValue in removed_strategies]
        #TODO: would be better to bundle into a single tx (effectively nets swaps)
        execution_response = asyncio.run(safe_gather(trades+unwinds))

        logging.getLogger('defillama_live').info(f'executed trades: {execution_response}')

    def make_trade(self, strategy_index, amount_usd: float):
        # convert amount to token value and on-chain swap format
        if amount_usd == 0:
            return

        price, decimals = 1, 18
        raise NotImplementedError('need to get price and decimals from somewhere')
        amount = (amount_usd / price) * (10 ** decimals)

        data_message = {'tradeType': 'MARKET',
                        'orderPrice': 0,
                        'orderAmount': amount,
                        'dynasetVaultId': self.vault_id,
                        'strategy_index': strategy_index,
                        'action': 'deposit' if amount > 0 else 'withdraw',
                        'unwrap': True  # swap from/to base_asset for now
                        }

        header = {'accept': 'application/json',
                  'signer': '1',
                  'signature': '1',
                  'msg': '0',
                  'Content-Type': 'application/json'}

        return requests.post('https://automated-execution-server-v2.dev.singularitydao.ai/execution-server/orders',
                            json=data_message,
                            headers=header
                            )

