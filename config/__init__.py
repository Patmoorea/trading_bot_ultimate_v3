# Version 2025-05-25 00:57:32
from decimal import Decimal
from datetime import datetime
from typing import Dict

class Config:
    # Paramètres temporels
    CURRENT_USER = "Patmoorea"
    TIMESTAMP = datetime(2025, 5, 25, 0, 57, 32)
    
    # Paramètres de risque
    RISK_PERCENTAGE = Decimal('0.02')
    MAX_POSITION_SIZE = Decimal('0.1')
    MAX_DRAWDOWN = Decimal('0.1')
    
    # Paramètres de circuit breakers
    LIQUIDITY_MIN_DEPTH = Decimal('10.0')
    CRASH_THRESHOLD = Decimal('0.1')
    VOLATILITY_BASE = Decimal('0.02')
    
    # Timeframes
    MONITORING_INTERVAL = 300

    @staticmethod
    def get_risk_params() -> Dict:
        return {
            'risk_percentage': Config.RISK_PERCENTAGE,
            'max_position_size': Config.MAX_POSITION_SIZE,
            'max_drawdown': Config.MAX_DRAWDOWN
        }

    @staticmethod
    def get_current_config() -> Dict:
        return {
            'user': Config.CURRENT_USER,
            'timestamp': Config.TIMESTAMP,
            'risk_params': Config.get_risk_params()
        }
