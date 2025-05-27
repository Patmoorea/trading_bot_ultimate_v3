"""
Configuration for arbitrage tests
Last updated: 2025-05-27 16:18:32 UTC by Patmoorea
"""

EXCHANGE_PAIRS = {
    'binance': ['BTC/USDC', 'ETH/USDC', 'BNB/USDC'],  # Binance utilise USDC
    'bingx': ['BTC/USDT', 'ETH/USDT'],    # BingX utilise USDT
}

TEST_TIMESTAMPS = {
    'current': 1811862312000,  # 2025-05-27 16:18:32 UTC
    'past': 1811862012000,     # 2025-05-27 16:13:32 UTC
    'future': 1811862612000    # 2025-05-27 16:23:32 UTC
}

SYSTEM_CONFIG = {
    'user': 'Patmoorea',
    'hardware': 'Apple M4',
    'os': 'macOS 15.3.2'
}
