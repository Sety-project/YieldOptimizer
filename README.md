# midfreq

ML library for mid-frequency trading, and DeFi yield portfolio optimization

## Architecture
### 1) Data collection (./scrappers)
#### a) ./scrappers/defillama_history (API to defillama)
Implements objects that lists all DefiLama pools satisfying some criteria:
- a set of whitelisted protocols (input in whitelist.yaml)
- a set of whitelisted underlyings (hard coded in code)
- pool criteria (e.g. Tvl, chain, etc), (hard coded in code)

It then collects data from Defillama, and adjusts APY (eg Haircut for reward tokens, add underlying APY, optionnaly assumes repegging and converts that into a yield).

Currently, it has 5 derived classes, but adding a class takes a few minutes:
- DynLst (LSTs on Ethereum)
- DynYieldE (stable on Ethereum)
- DynYieldB (stables on BSC)
- DynYieldA (stables on Arbitrum)
- DynYieldA (stables on Arbitrum)
- DynYieldBTCE (WBTC on Ethereum)
- DiscoveryDefiLlama: all pools satisfying flexible criteria

#### b) ./scrappers/binance_history (binance data download) (DEPRECATED)
- bash scripts to download data from https://data.binance.vision/
### 2) Feature engineering and modelling (./research)
- ResearchEngine class reads data from files, and provides methods to generate derived features (emwa, hvol, etc) as well as derived labels (multi horizon)
- provides methods to train and cross validate models, including ability to train one model across several instruments
- support most models in sklearn, as well as some custom models
### 4) bet sizing (./strategies)
#### a) ./strategies/cta_strategy.py (DEPRECATED)
attempts to learn large moves and trade along predict_proba
#### b) ./strategies/vault_rebalancing.py 
performs portfolio optimization w/ transaction costs assuming a holding horizon. This performs a contrained convex optimization at each time step.
### 5) Backtesting (./strategies)
#### a) ./strategies/cta_backtest.py (DEPRECATED)
backtests cta_strategy.py instrument by instrument, and compute metrics
#### b) ./strategies/vault_backtest.py 
backtests vault_rebalancing.py
### 6) misc
- defillama data and metadata lands in scrappers/defillama_history/data
- ./config has a big yaml input parametrizing prediction, betasizing, backtest etc..
# guide
- to install, run `pip install -r requirements.txt`
- to download defillama history, run `python ./scrappers/defillama_history/defillama.py defillama [any class name derived from FilteredDefiLama] [whitelist.yaml path]`<br>
This will generate files with history: ./scrappers/defillama_history/data/[class name]/[poolId].csv<br>
and also one summary file: ./scrappers/defillama_history/data/[class name]_pool_metadata.csv
- Once this is done, run `top n ever.ipynb` to generate a list of top APY pools. Copy the last cell as input_data/selected_instruments in the ./config/[vault_name].yaml file for the vault you are optimizing.
- then run `python main.py`[command] [config/vault_name.yaml].<br>
Command `backtest` will run backtests, and `grid` will run several times with different hyperparameters (define grid in `__main__` code)
- This will generate full backtest results in ./logs/[vault_name]/[param values].csv <br>
and a summary of all backtests in ./logs/[vault_name]/grid.csv
- Open `DynBacktests-research.ipynb` to analyze backtest results and graph perf trajectories, hyperparameters sensitivities etc...