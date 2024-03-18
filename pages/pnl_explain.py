import asyncio
import html
import os
import sys
from datetime import timedelta, date, datetime, timezone

import matplotlib
import pandas as pd
import streamlit as st
import yaml

from utils.async_utils import safe_gather
from utils.postgres import SqlApi, CsvDB
from plex.debank_api import DebankAPI
from utils.streamlit_utils import authentification_sidebar, load_parameters

assert (sys.version_info >= (3, 10)), "Please use Python 3.10 or higher"

pd.options.mode.chained_assignment = None
st.session_state.parameters = load_parameters()

st.session_state.database = CsvDB()
authentification_sidebar()

snapshot_tab, risk_tab, plex_tab = st.tabs(
    ["snapshot", "risk", "pnl explain"])

if 'stage' not in st.session_state:
    st.session_state.stage = 0

if 'my_addresses' not in st.secrets:
    addresses = st.sidebar.text_area("addresses", help='Enter multiple strings on separate lines').split('\n')
else:
    addresses = st.secrets['my_addresses']
addresses = [address for address in addresses if address[:2] == "0x"]

with snapshot_tab:
    with st.form("snapshot_form"):
        if st.form_submit_button("take snapshot"):
            if st.session_state.authentification == 'verified':
                with st.spinner(f"Taking snapshots"):
                    debank_key = st.text_input("debank key",
                                                  value=st.secrets['debank'] if 'debank' in st.secrets else '',
                                                  help="you think i am going to pay for you?")
                    obj = DebankAPI(debank_key)

                    async def position_snapshots() -> dict[str, pd.DataFrame]:
                        # only update once every 'update_frequency' minutes
                        last_updated = await safe_gather(
                            [st.session_state.database.last_updated(address)
                              for address in addresses],
                            n=st.session_state.parameters['input_data']['async']['gather_limit'])
                        max_updated = datetime.now(tz=timezone.utc) - timedelta(minutes=st.session_state.parameters['plex']['update_frequency'])
                        addresses_to_refresh = [address
                                  for address, last_update in zip(addresses, last_updated)
                                  if last_update < max_updated]
                        if not addresses_to_refresh:
                            st.warning(f"We only update once every {st.session_state.parameters['plex']['update_frequency']} minutes. No addresses to refresh")
                            return {}

                        # fetch snapshots
                        results = await safe_gather(
                            [obj.fetch_position_snapshot(address)
                              for address in addresses_to_refresh],
                            n=st.session_state.parameters['input_data']['async']['gather_limit'])

                        # write to db
                        await safe_gather(
                            [st.session_state.database.insert_snapshot(result, address)
                              for address, result in zip(addresses_to_refresh, results)],
                            n=st.session_state.parameters['input_data']['async']['gather_limit'])

                        return dict(zip(addresses, results))

                    snapshot_by_address = asyncio.run(position_snapshots())
                    for address, snapshot in snapshot_by_address.items():
                        st.write(address)
                        st.dataframe(snapshot)
                    st.session_state.snapshots = snapshot_by_address
                    st.session_state.stage = 1
            else:
                st.warning(
                    html.unescape(
                        'chat https://t.me/Pronoia_Bot, then enter your tg handle in the sidebar to get access'
                    )
                )

with risk_tab:
    snapshot = None
