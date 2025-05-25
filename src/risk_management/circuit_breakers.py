import numpy as np
from typing import Dict, Any

class CircuitBreakers:
    def __init__(self, config: Dict[str, Any]):
        self.thresholds = {
            'market_crash': config.get('market_crash_threshold', -0.15),  # -15%
            'liquidity_shock': config.get('liquidity_shock_threshold', 3.0),  # 3x normal spread
            'black_swan': config.get('black_swan_threshold', -0.30)  # -30%
        }
        self.state = {
            'trading_enabled': True,
            'last_trigger': None
        }
    
    def check_market_conditions(self, market_data: Dict[str, Any]):
        """Evaluate all circuit breaker conditions"""
        triggers = {
            'market_crash': self._check_market_crash(market_data['returns']),
            'liquidity_shock': self._check_liquidity_shock(market_data['spreads']),
            'black_swan': self._check_black_swan(market_data['prices'])
        }
        
        if any(triggers.values()):
            trigger = next(k for k, v in triggers.items() if v)
            self._activate_circuit_breaker(trigger)
            return False, trigger
        return True, None
    
    def _check_market_crash(self, returns):
        return np.any(returns < self.thresholds['market_crash'])
    
    def _check_liquidity_shock(self, spreads):
        median_spread = np.median(spreads)
        return np.any(spreads > median_spread * self.thresholds['liquidity_shock'])
    
    def _check_black_swan(self, prices):
        max_drawdown = (np.max(prices) - np.min(prices)) / np.max(prices)
        return max_drawdown > self.thresholds['black_swan']
    
    def _activate_circuit_breaker(self, trigger):
        self.state.update({
            'trading_enabled': False,
            'last_trigger': trigger
        })
    
    def reset(self):
        self.state.update({
            'trading_enabled': True,
            'last_trigger': None
        })
