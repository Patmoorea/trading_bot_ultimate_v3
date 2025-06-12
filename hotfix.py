from modules.arbitrage_engine import ArbitrageEngine

# Monkey patching sécurisé
if not hasattr(ArbitrageEngine, 'check_opportunities_v2'):
    def _check_opportunities_v2(self):
        return self.find_usdc_arbitrage(min_spread=0.008) if hasattr(self, "find_usdc_arbitrage") else []
    
    ArbitrageEngine.check_opportunities_v2 = _check_opportunities_v2
