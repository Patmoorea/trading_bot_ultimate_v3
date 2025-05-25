def calculate_net_arbitrage(spread):
    """Prend en compte les frais de trading"""
    fees = {"binance": 0.001, "kraken": 0.0016, "huobi": 0.002}  # 0.1%
    return spread - sum(fees.values())
