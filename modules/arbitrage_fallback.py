from modules.arbitrage_engine import ArbitrageEngine as OriginalEngine

class ArbitrageEngine(OriginalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'check_opportunities_v2'):
            self.check_opportunities_v2 = self._fallback_check
        if not hasattr(self, '_calculate_spreads'):
            self._calculate_spreads = self._dummy_calculate_spreads

    def _fallback_check(self):
        return []

    def _dummy_calculate_spreads(self, pair):
        print(f"Avertissement: Calcul de spread factice pour {pair}")
        return {'min': 0, 'max': 0}
