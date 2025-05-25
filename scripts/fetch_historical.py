import pandas as pd
from binance.spot import Spot

def get_historical_data(symbol, interval, limit):
    client = Spot()
    klines = client.klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df.to_csv(f'data/historical/{symbol}_{interval}.csv', index=False)

if __name__ == "__main__":
    get_historical_data('BTCUSDT', '1h', 1000)
