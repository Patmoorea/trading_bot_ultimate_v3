from binance.client import Client
import pandas as pd
import os
from tqdm import tqdm
import time

def fetch_with_progress(client, symbol, interval, start_date):
    klines = []
    pbar = tqdm(desc="Téléchargement des données")
    
    while True:
        new_klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_date
        )
        if not new_klines:
            break
        klines.extend(new_klines)
        pbar.update(len(new_klines))
        start_date = pd.to_datetime(new_klines[-1][0], unit='ms') + pd.Timedelta(1, unit='ms')
        time.sleep(0.2)  # Respect rate limits
    
    pbar.close()
    return klines

def fetch_historical_data(symbol='BTCUSDT', interval='1h', start_date='2020-01-01'):
    client = Client()
    os.makedirs('data/historical', exist_ok=True)
    
    print(f"Début du téléchargement des données {symbol} {interval} depuis {start_date}...")
    klines = fetch_with_progress(client, symbol, interval, start_date)
    
    if not klines:
        print("Aucune donnée récupérée!")
        return

    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore']
    
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    
    filename = f"data/historical/{symbol.lower()}_{interval}.csv"
    df[['timestamp'] + numeric_cols].to_csv(filename, index=False)
    print(f"\nDonnées sauvegardées dans {filename} ({len(df)} lignes)")

if __name__ == '__main__':
    fetch_historical_data()
