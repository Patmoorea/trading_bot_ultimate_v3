from decimal import Decimal
from datetime import datetime
from typing import Dict

class RiskConfig:
    # Paramètres temporels
    CURRENT_USER = "Patmoorea"
    TIMESTAMP = datetime(2025, 5, 25, 0, 56, 30)
    
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
            'risk_percentage': RiskConfig.RISK_PERCENTAGE,
            'max_position_size': RiskConfig.MAX_POSITION_SIZE,
            'max_drawdown': RiskConfig.MAX_DRAWDOWN
        }
    
    @staticmethod
    def get_current_config() -> Dict:
        return {
            'user': RiskConfig.CURRENT_USER,
            'timestamp': RiskConfig.TIMESTAMP,
            'risk_params': RiskConfig.get_risk_params()
        }

# Alias pour la compatibilité
Config = RiskConfig
