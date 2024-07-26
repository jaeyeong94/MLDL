import pandas as pd

features = [
    'mid_price',
    'best_ask',
    'best_bid',
    'spread',
    'spread_num',
    'depth_ask',
    'depth_bid',
    'a10',
    'b10',
    'obi',
    'rise_ratio'
]


def parse_orderbook(row):
    asks = pd.DataFrame((eval(row['asks'])))
    bids = pd.DataFrame((eval(row['bids'])))

    best_ask = asks['price'].min()
    best_bid = bids['price'].max()

    spread_num = best_ask - best_bid

    depth_ask = asks['size'].sum()
    depth_bid = bids['size'].sum()
    a10 = asks['size'].iloc[:10].sum()
    b10 = bids['size'].iloc[:10].sum()
    obi = (depth_ask - depth_bid) / (depth_ask + depth_bid)

    rise_ratio = (a10 + b10) / (depth_ask + depth_bid)

    return pd.Series([row['mid_price'], best_ask, best_bid, row['spread'], spread_num, depth_ask, depth_bid, a10, b10, obi, rise_ratio])
