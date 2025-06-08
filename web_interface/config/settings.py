import os
from datetime import timedelta

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'trading_bot_ultimate_2025')
    
    # WebSocket
    WS_PING_INTERVAL = 25
    WS_PING_TIMEOUT = 120
    
    # Trading
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
    MAX_WEBSOCKETS = 12
    BUFFER_SIZE = 1000
    
    # Performance
    LATENCY_THRESHOLD = 50  # ms
    UPDATE_INTERVAL = 1000  # ms
    
    # Security
    SESSION_LIFETIME = timedelta(hours=12)
    
    # Features
    FEATURES = {
        'news_integration': True,
        'voice_recognition': True,
        'telegram_alerts': True,
        'ml_optimization': True
    }
