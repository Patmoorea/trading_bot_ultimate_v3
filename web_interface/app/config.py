import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv('/Users/patricejourdan/Desktop/trading_bot_ultimate/.env')

class Config:
    # Server Configuration
    PORT = 5001
    HOST = '0.0.0.0'
    DEBUG = True
    
    # User Configuration
    CURRENT_USER = "Patmoorea"
    
    # Trading Configuration
    TRADING_MODE = "BUY_ONLY"
    BASE_CURRENCY = "USDC"
    PAIRS = ["BTC/USDC", "ETH/USDC", "BNB/USDC", "SOL/USDC", "XRP/USDC"]
    
    # Timeframes
    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    # Risk Management
    MAX_POSITION_SIZE = 10000  # USDC
    DAILY_STOP_LOSS = 0.02    # 2%
    MAX_DRAWDOWN = 0.05       # 5%
    
    # Telegram Configuration (depuis .env)
    TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true'
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
