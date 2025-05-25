import ccxt
import numpy as np


def find_temporal_opportunities():
    """Détecte les décalages entre exchanges"""
    exchanges = {
        "binance": ccxt.binance(),
        "kraken": ccxt.kraken(),
        "huobi": ccxt.huobi(),
    }

    opportunities = []
    for pair in ["BTC/USDC", "ETH/USDC"]:
        prices = []
        for name, exchange in exchanges.items():
            try:
                ticker = exchange.fetch_ticker(pair)
                prices.append((name, ticker["last"]))
            except BaseException:
                continue

        if len(prices) > 1:
            max_diff = max(p[1] for p in prices) - min(p[1] for p in prices)
            spread = max_diff / min(p[1] for p in prices)
            if spread > 0.0015:  # 0.15%
                opportunities.append(
                    {"pair": pair,
                     "spread": f"{spread*100:.2f}%",
                     "exchanges": prices}
                )

    return opportunities
