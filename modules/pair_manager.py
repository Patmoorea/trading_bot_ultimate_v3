import os

from dotenv import load_dotenv

load_dotenv()


def get_trading_pairs():
    """Récupère les paires depuis .env avec fallback"""
    pairs = os.getenv("TRADING_PAIRS", "BTC,ETH").split(",")
    if os.getenv("BINANCE_USDC_MODE") == "true":
        return [f"{p}/USDC" for p in pairs]
    return [f"{p}/USDT" for p in pairs]
