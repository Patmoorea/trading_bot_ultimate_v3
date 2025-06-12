import pandas as pd
df = pd.read_csv('data/historical/btc_usdt_1h_ultraclean.csv')
df[['timestamp','open','high','low','close','volume']].to_csv('data/historical/btc_usdt_1h_optimized.csv', index=False)
