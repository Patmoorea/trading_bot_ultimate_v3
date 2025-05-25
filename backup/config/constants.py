"""
Constantes de trading
"""
class Constants:
    # Timeframes
    TIMEFRAMES = ['15m', '1h', '4h', '1d']
    
    # Indicateurs techniques
    INDICATORS = {
        'trend': ['ichimoku', 'supertrend', 'vwma'],
        'momentum': ['rsi', 'stoch_rsi', 'macd'],
        'volatility': ['atr', 'bb_width', 'keltner'],
        'volume': ['obv', 'vwap', 'accumulation']
    }
    
    # Paramètres de risque
    RISK_PARAMS = {
        'max_drawdown': 0.05,
        'daily_stop_loss': 0.02,
        'position_sizing': 'volatility_based'
    }
