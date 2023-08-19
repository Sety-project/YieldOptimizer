# midfreq

ML library for mid-frequency trading, and DeFi yield portfolio optimization

## Architecture
### 1) Data collection (./scrappers)
#### a) ./scrappers/defillama_history (API to defillama)
- finds all pools for given protocol, with given underlyings
- Collects data from Defillama, and adjusts APY
#### b) ./scrappers/binance_history (API to binance)
- bash scripts to download data from https://data.binance.vision/
### 2) Feature engineering and modelling (./research)
- ResearchEngine class reads data from files, and provides methods to generate derived features (emwa, hvol, etc) as well as derived labels (multi horizon)
- provides methods to train and cross validate models, inclusing ability to train one model across several instruments
- support most models in sklearn, as well as some custom models
### 4) bet sizing (./strategies)
#### a) ./strategies/cta_strategy.py attempts to learn large moves and trade along predict_proba
#### b) ./strategies/vault_rebalancing.py performs portfolio optimization w/ transaction costs assuming a holding horizon
### 5) Backtesting (./strategies)
#### a) ./strategies/cta_backtest.py backtests cta_strategy.py instrument by instrument, and compute metrics
#### b) ./strategies/vault_backtest.py backtests vault_rebalancing.py
### 6) misc
- defillama data and metadata lands in scrappers/defillama_history/data
- ./config has a big json input parametrizing all the above
### 7) guides
- to install, run `pip install -r requirements.txt`
- to run, run `python main.py`[command] [config file path]
