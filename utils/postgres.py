import sys
from datetime import datetime
from datetime import timezone

import pandas as pd
import streamlit as st
from sqlalchemy import Float, DateTime, Connection, Engine
from sqlalchemy import create_engine, inspect, MetaData, Table
from sqlalchemy import text as sql_text

from utils.async_utils import async_wrap


class SqlApi():
    def __init__(self, database: str, schema=None):
        if schema is None:
            self.schema = {'date': DateTime(timezone=True),
                           'haircut_apy': Float,
                           'apy': Float,
                           'apyReward': Float,
                           'il': Float,
                           'tvl': Float}
        self.engine: Engine = create_engine(database)
        self.connection: Connection = self.engine.connect()
        self.tables: list = self.list_tables()

    #@st.cache_data
    def read_sql_query(_self, query: str, **kwargs):
        '''st.cache_data doesn't support async functions so we have this wrapper
        the leading underscore is also to accommodate st.cache_data'''
        return pd.read_sql_query(sql_text(query), con=_self.connection, **kwargs)

    async def read(self, instrument: str=None):
        query = f'''SELECT * FROM \"{instrument}\" WHERE date >= \'{datetime(2023, 2, 20).strftime("%Y-%m-%d")}\'::date ORDER BY date ASC;'''
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
            result = f"Added {new_data.shape[0]} rows to {pool}"
        else:
            new_data = data
            result = f"Created {pool}"

        new_data['updated'] = datetime.now(timezone.utc)
        await async_wrap(new_data.to_sql)(name=pool, con=self.engine, if_exists='append', index=False, dtype=self.schema)
        return result

    async def read_metadata(self):
        query = f'''SELECT pool, updated FROM metadata;'''
        result = await async_wrap(self.read_sql_query)(query, index_col='pool')
        return result

    async def write_metadata(self, data: pd.DataFrame):
        data['updated'] = datetime.now(timezone.utc)
        await async_wrap(data.to_sql)(name='metadata', con=self.engine, if_exists='replace', index=False,
                                          dtype=self.schema)
        return "updated metadata"

    async def last_updated(self, metadata: dict = None):
        if 'metadata' in self.tables:
            prev_metadata = await self.read_metadata()
            return prev_metadata[metadata["pool"]].squeeze()
        elif metadata['name'] in self.tables:
            query = f'''SELECT MAX(updated) FROM \"{metadata['name']}\";'''
            result = await async_wrap(pd.read_sql_query)(sql_text(query), con=self.connection)
            return result.squeeze()
        else:
            return None

    def delete(self, tables: list[str] = None):
        metadata = MetaData()
        metadata.reflect(self.engine)
        if tables is None:
            metadata.drop_all(bind=self.engine)
        else:
            metadata.drop_all(bind=self.engine, tables=list(map(metadata.tables.get, tables)))

    def list_tables(self):
        meta = MetaData()
        meta.reflect(bind=self.engine)
        result = [tab for tab in meta.tables if tab!='metadata']
        return result


if __name__ == '__main__':
    sql_obj = SqlApi(st.secrets[sys.argv[2]])
    if sys.argv[1] == "delete":
        sql_obj.delete(sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "read":
        sql_obj.read(sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "list_tables":
        print(sql_obj.list_tables())
    elif sys.argv[1] == "write":
        df = pd.DataFrame({key['field']: range(5) for key in sql_obj.schema})
        sql_obj.write("Arbitrum_aave-v3_USDC")
    else:
        print("invalid argument")