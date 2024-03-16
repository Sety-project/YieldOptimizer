import logging
import sys
import typing
from datetime import datetime
from datetime import timezone

import pandas as pd
import streamlit as st
from sqlalchemy import Float, DateTime, Connection, Engine, String
from sqlalchemy import MetaData
from sqlalchemy import text as sql_text

from utils.async_utils import async_wrap
from utils.io_utils import profile_all

#@profile_all
class SqlApi:
    def __init__(self, name: str, pool_size: int, max_overflow: int, pool_recycle: int):
        self.pool_schema = {'date': DateTime(timezone=True),
                            'haircut_apy': Float,
                            'apy': Float,
                            'apyReward': Float,
                            'il': Float,
                            'tvl': Float,
                            'updated': DateTime(timezone=True)}
        self.plex_schema = {'chain': String(16),
                            'protocol': String(32),
                            # 'description': portfolio_item['detail']['description'],
                            'hold_mode': String(16),
                            'type': String(16),
                            'asset': String(32),
                            'amount': Float,
                            'price': Float,
                            'value': Float,
                            'updated': DateTime(timezone=True)}
        self.engine = st.connection(name, type="sql", autocommit=True,
                                    pool_size=pool_size, max_overflow=max_overflow, pool_recycle=pool_recycle)
        # self.engine: Engine = create_engine(database,
        #                                     pool_size=pool_size,
        #                                     max_overflow=max_overflow,
        #                                     pool_recycle=pool_recycle)
        self.tables: list = self.list_tables()

    #@st.cache_data
    def read_sql_query(_self, query: str, **kwargs):
        '''st.cache_data doesn't support async functions so we have this wrapper
        the leading underscore is also to accommodate st.cache_data'''
        with _self.engine.connect() as con:
            return pd.read_sql_query(sql_text(query), con=con, **kwargs)

    async def read_one(self, instrument: str):
        query = f'''SELECT * FROM \"{instrument}\" ORDER BY date ASC;'''
        result = await async_wrap(self.read_sql_query)(query, index_col='date')
        result.index = [pd.to_datetime(x, unit='ns', utc=True) for x in result.index]
        return result[~result.index.duplicated()]

    async def write(self, data: pd.DataFrame, pool: str) -> str:
        '''write to db
         if table exists, only write new data
         add 'updated' column'''
        if pool in self.list_tables():
            existing_dates = self.read_sql_query(f"""SELECT date FROM \"{pool}\"""")
            new_data = data[~data['date'].isin(existing_dates['date'].apply(lambda t: pd.to_datetime(t, infer_datetime_format=True, utc=True, unit='ns', errors='coerce')))]
            result = f"Added: {new_data.shape[0]} rows to {pool}"
        else:
            new_data = data
            result = f"Created: {pool}"

        if not new_data.empty:
            new_data['updated'] = datetime.now(timezone.utc)
            with self.engine.connect() as connection:
                await async_wrap(new_data.to_sql)(name=pool, con=connection, if_exists='append', index=False, dtype=self.pool_schema)
        else:
            result = f"No new data: {pool}"
        return result

    async def last_updated(self, name: str) -> typing.Union[datetime, None]:
        if name in self.tables:
            query = f'''SELECT MAX(updated) FROM \"{name}\";'''
            result = await async_wrap(self.read_sql_query)(query)
            return result.squeeze().replace(tzinfo=timezone.utc)
        else:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    def is_whitelisted(self, tg_username: str) -> bool:
        if 'interactions' not in self.tables:
            with self.engine.session as connection:
                connection.execute(sql_text(
                    "CREATE TABLE interactions (username VARCHAR(255), timestamp TIMESTAMP WITH TIME ZONE, message VARCHAR(255));")
                )
                connection.commit()
            return False
        query = f'''SELECT username FROM interactions WHERE username = '{tg_username}';'''
        result = self.read_sql_query(query)
        return not result.empty

    def record_interaction(self, tg_username: str, timestamp: datetime, message: str):
        with self.engine.session as connection:
            connection.execute(sql_text(f"INSERT INTO interactions (username, timestamp, message) VALUES ('{tg_username}', '{timestamp}', '{message}')"))
            connection.commit()

    async def insert_snapshot(self, snapshot: pd.DataFrame, address: str) -> None:
        # if address not in self.tables:
        #     with self.engine.session as connection:
        #         connection.execute(sql_text(
        #             f"CREATE TABLE {address} (username VARCHAR(255), timestamp TIMESTAMP WITH TIME ZONE, message VARCHAR(255));")
        #         )
        #         connection.commit()
        with self.engine.connect() as connection:
            await async_wrap(snapshot.to_sql)(name=address, con=connection, if_exists='append', index=False, dtype=self.plex_schema, method='multi')


    def delete(self, tables: list[str] = None):
        metadata = MetaData()
        metadata.reflect(self.engine.engine)
        if tables is None:
            metadata.drop_all(bind=self.engine.engine)
        else:
            metadata.drop_all(bind=self.engine.engine, tables=list(map(metadata.tables.get, tables)))

    def list_tables(self):
        meta = MetaData()
        meta.reflect(bind=self.engine.engine)
        return list(meta.tables)

if __name__ == '__main__':
    sql_obj = SqlApi(st.secrets[sys.argv[2]])
    if sys.argv[1] == "delete":
        sql_obj.delete(sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "read":
        sql_obj.read(sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "list_tables":
        print(sql_obj.list_tables())
    elif sys.argv[1] == "write":
        df = pd.DataFrame({key['field']: range(5) for key in sql_obj.pool_schema})
        sql_obj.write("Arbitrum_aave-v3_USDC")
    else:
        print("invalid argument")