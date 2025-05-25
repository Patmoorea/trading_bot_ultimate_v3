
class MarketAdaptivePlanner:
    def __init__(self):
        self.memory_size = 1000  # candles stored
    
    def generate_plan(self, market_data):
        analysis = {
            'trend': self._detect_trend(market_data),
            'volatility': self._calc_volatility(market_data),
            'liquidity': self._assess_liquidity(market_data)
        }
        
        if analysis['volatility'] > 0.05:
            return {'strategy': 'scalping', 'timeframe': '5m'}
        else:
            return {'strategy': 'swing', 'timeframe': '4h'}
