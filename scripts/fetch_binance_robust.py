import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import time

def safe_fetch(url, max_retries=3):
    for _ in range(max_retries):
        try:
            r = requests.get(url, timeout=10)
            return r.json()
        except Exception as e:
            print(f"Erreur, réessai dans 5s... ({str(e)})")
            time.sleep(5)
    return None

def fetch_history(symbol, interval, start_date, end_date=None):
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_date = start_date
    
    while True:
        url = f"{base_url}?symbol={symbol}&interval={interval}&startTime={int(current_date.timestamp()*1000)}"
        if end_date:
            url += f"&endTime={int(end_date.timestamp()*1000)}"
        url += "&limit=1000"
        
        data = safe_fetch(url)
        if not data:
            break
            
        all_data.extend(data)
        if len(data) < 1000:
            break
            
        current_date = datetime.fromtimestamp(data[-1][0]/1000) + timedelta(seconds=1)
        time.sleep(0.1)
    
    return all_data

if __name__ == '__main__':
    os.makedirs('data/historical', exist_ok=True)
    
    try:
        print("Téléchargement des 1000 dernières bougies...")
        data = fetch_history('BTCUSDT', '1h', datetime.now()-timedelta(days=60))
        
        if data:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count',
                'taker_buy_volume', 'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.to_csv('data/historical/btc_usdt_1h.csv', index=False)
            print(f"✅ {len(df)} points sauvegardés")
        else:
            print("❌ Échec du téléchargement")
    except Exception as e:
        print(f"Erreur critique: {str(e)}")
