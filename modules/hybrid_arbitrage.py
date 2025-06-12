from modules.arbitrage_engine import ClassicArbitrage
from modules._arbitrage_fallback import safe_find_arbitrage

class HybridArbitrage(ClassicArbitrage):
    """Solution hybride définitive"""
    
    def calculate(self):
        base = super().calculate()
        fallback = safe_find_arbitrage()
        return {
            'base': base,
            'fallback': fallback,
            'combined': {**base, **fallback}
        }
