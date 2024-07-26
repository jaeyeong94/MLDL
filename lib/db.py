import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

exchange_host = {
    'vertex-perp': os.environ.get('POSTGRES_HOST2'),
    'okx-perp': os.environ.get('POSTGRES_HOST2'),
    'okx': os.environ.get('POSTGRES_HOST2'),
    'binance-perp': os.environ.get('POSTGRES_HOST2'),
    'binance': os.environ.get('POSTGRES_HOST2'),
    'coinbase': os.environ.get('POSTGRES_HOST2'),
    'bybit-perp': os.environ.get('POSTGRES_HOST'),
    'bybit': os.environ.get('POSTGRES_HOST'),
    'upbit': os.environ.get('POSTGRES_HOST'),
    'bithumb': os.environ.get('POSTGRES_HOST'),
    'hyperliquid-perp': os.environ.get('POSTGRES_HOST')
}


def get_query(query_string, exchange_name, params=None, chunk_size=10000) -> pd.DataFrame:
    host = get_host_by_exchange(exchange_name)
    engine = create_engine(f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{host}:{os.environ.get('POSTGRES_PORT')}/{os.environ.get('POSTGRES_DATABASE')}")
    chunks = []

    for i, chunk in enumerate(pd.read_sql_query(query_string, engine, params=params, chunksize=chunk_size)):
        print(f'{exchange_name} - {i} chunk')
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return df


def get_query_to_file(query_string, exchange_name, params=None, chunk_size=10000, dir_name='chunk') -> None:
    host = get_host_by_exchange(exchange_name)
    engine = create_engine(f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{host}:{os.environ.get('POSTGRES_PORT')}/{os.environ.get('POSTGRES_DATABASE')}")

    for i, chunk in enumerate(pd.read_sql_query(query_string, engine, params=params, chunksize=chunk_size)):
        print(f'{exchange_name} - {i} chunk')
        chunk.to_csv(f'{dir_name}/{exchange_name}_orderbook_chunk_{i}.csv', index=False)

    print('----------------- File Save Done')
    return None


def get_host_by_exchange(exchange_name):
    if exchange_name in exchange_host:
        return exchange_host[exchange_name]
    else:
        raise Exception('Exchange not found in exchange_host')
