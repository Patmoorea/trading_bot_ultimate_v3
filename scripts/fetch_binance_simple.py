import pandas as pd
import requests
import os

def get_binance_data(symbol='BTCUSDT', interval='1h', limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count',
        'taker_buy_volume', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

if __name__ == '__main__':
    os.makedirs('data/historical', exist_ok=True)
    df = get_binance_data(limit=500)  # Test avec seulement 500 points
    df.to_csv('data/historical/btc_usdt_1h.csv', index=False)
    print(f"Données sauvegardées (500 derniers points)")
