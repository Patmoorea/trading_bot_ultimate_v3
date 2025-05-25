"""
Configuration des paths
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class Paths:
    DATA = BASE_DIR / 'data'
    LOGS = BASE_DIR / 'logs'
    MODELS = BASE_DIR / 'models'
    
    MARKET_DATA = DATA / 'market'
    SIGNALS = DATA / 'signals'
    BACKTEST = DATA / 'backtest'
    NEWS = DATA / 'news'
