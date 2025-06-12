def detect_whale_activity(pair):
    exchange = ccxt.binance()
    trades = exchange.fetch_trades(pair, limit=100)
    large_trades = [t for t in trades if t["amount"] > 10]  # >10 BTC
    return len(large_trades) > 5  # Alerte si >5 gros trades récents
