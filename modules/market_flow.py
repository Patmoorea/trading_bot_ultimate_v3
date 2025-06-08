def detect_whale_orders(pair, window=100):
    """Détecte les gros ordres récents"""
    exchange = ccxt.binance()
    trades = exchange.fetch_trades(pair, limit=window)

    whale_threshold = {
        "BTC/USDC": 10,  # 10 BTC
        "ETH/USDC": 50,  # 50 ETH
    }.get(
        pair, 10000
    )  # 10k USD par défaut

    return [t for t in trades if t["amount"] >= whale_threshold]
