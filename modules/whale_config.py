WHALE_THRESHOLDS = {
    "BTC/USDC": 10,  # 10 BTC
    "ETH/USDC": 50,  # 50 ETH
    "SOL/USDC": 500,  # 500 SOL
    "DEFAULT": 10000,  # 10k USD
}


def get_threshold(pair):
    return WHALE_THRESHOLDS.get(pair, WHALE_THRESHOLDS["DEFAULT"])
