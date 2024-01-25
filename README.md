# https://poolhopper.streamlit.app/

Streamlit app that backtests a DeFi yield optimization strategy. Uses a yield predictor and is mindful of slippage, gaz, yield dilution.

More details on https://medium.com/@nononymous/poolhopper-7fb6cf18182b

## Architecture
### 1) Data collection (./scrappers)
#### a) ./scrappers/defillama_history (API to defillama)
First, filter DefiLama pools satisfying some criteria:
- a set of whitelisted protocols
- a set of whitelisted underlyings
- pool criteria (e.g. Tvl, chain, etc)

Then:
 - collects data from Defillama.
 - adjusts APY (eg Haircut rewards, add underlying APY (eg stETH), optionally assumes repegging and converts that into a yield).
 - feeds a Postgres DB, either local or hosted by Aiden (name specified in params.yaml and auth in secrets.toml)
### 2) Feature engineering and modelling (./research)
- ResearchEngine class reads data from database, and provides methods to generate derived features (emwa, hvol, etc) as well as derived labels (multi horizon)
- provides methods to train and cross validate models.
- support most models in sklearn, as well as some custom models
### 3) bet sizing (./strategies/vault_rebalancing.py)
Performs portfolio optimization w/ transaction costs assuming a holding horizon. This performs a contrained convex optimization at each time step. All costs are annualized over am externally calibrated  horizon. 
### 4) Backtesting (./strategies/vault_backtest.py) 
Backtests vault_rebalancing.py, with one or several sets of parameters
### 5) configs (./config)
- params.yaml is big yaml input parametrizing prediction, bet sizing, backtest etc..
- whitelist.yaml is the default list of protocols.
- grid.yaml is the default grid of parameters to backtest
### 6) streamlit UI (./main.py)
- leverages postgres DB to store data (./utils/postgres.py)
- authentification through telegram bot (./utils/telegram_bot.py)
# guide
- to install, run `pip install -r requirements.txt`
- then run module streamlit `run main.py` to launch the streamlit app