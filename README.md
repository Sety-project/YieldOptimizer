# midfreq

ML library for mid-frequency trading, and DeFi yield portfolio optimization

## Architecture
### 1) Data collection (./scrappers)
#### a) ./scrappers/defillama_history (API to defillama)
Implements objects that lists all DefiLama pools satisfying some criteria:
- a set of whitelisted protocols (input in whitelist.yaml)
- a set of whitelisted underlyings (hard coded in code)
- pool criteria (e.g. Tvl, chain, etc), (hard coded in code)

It then:
 - collects data from Defillama
 - adjusts APY (eg Haircut for reward tokens, add underlying APY, optionnaly assumes repegging and converts that into a yield).
 - write to database (name specified in params.yaml and details in secrets.toml)
### 2) Feature engineering and modelling (./research)
- ResearchEngine class reads data from database, and provides methods to generate derived features (emwa, hvol, etc) as well as derived labels (multi horizon)
- provides methods to train and cross validate models, including ability to train one model across several instruments
- support most models in sklearn, as well as some custom models
### 3) bet sizing (./strategies/vault_rebalancing.py)
performs portfolio optimization w/ transaction costs assuming a holding horizon. This performs a contrained convex optimization at each time step.
### 4) Backtesting (./strategies/vault_backtest.py) 
backtests vault_rebalancing.py, with one of several sets of parameters
### 5) misc
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