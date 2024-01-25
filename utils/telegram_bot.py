import asyncio
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import requests
import telegram

def check_whitelist(tg_username: str) -> bool:
    if st.session_state.database.is_whitelisted(tg_username) or (tg_username in st.secrets.admins):
        st.session_state.database.record_interaction(tg_username, datetime.now(tz=timezone.utc), 'login')
        st.sidebar.success(f"Welcome back, {tg_username}")
        return True
    url = f"https://api.telegram.org/bot{st.secrets.telegram}/getUpdates"
    all_chats = requests.get(url).json()['result']
    latest_interaction = next(reversed([chat['message'] for chat in all_chats if chat['message']['from']['username'] == tg_username]), None)
    bot = telegram.Bot(token=st.secrets.telegram)
    if latest_interaction is not None:
        asyncio.run(bot.send_message(chat_id=latest_interaction['chat']['id'],
                                     text=f"ok {latest_interaction['from']['first_name']}, you're in"))
        st.session_state.database.record_interaction(tg_username, pd.to_datetime(latest_interaction['date'], unit='s', utc=True), latest_interaction['message']['text'])
        st.sidebar.success(f"Thanks for registering, {tg_username}")
        return True
    else:
        st.sidebar.warning("Chat with @Pronoia_Bot to get whitelisted")
        return False
