"""
DEMO FINALE - 100% compatible avec votre projet
"""
import sys
from pathlib import Path

# Configuration ABSOLUE du path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import DIRECT depuis votre module principal
from strategies.arbitrage.arbitrage import USDCArbitrage  # Chemin exact

class RiskExtension:
    """Extension qui n'altère pas votre code"""
    def __init__(self, arbitrage_core):
        self.core = arbitrage_core
    
    def validate(self, pair):
        """Méthode de validation autonome"""
        try:
            spread = self.core._calculate_spread(pair)
            return spread >= 0.002  # Seuil personnalisable
        except:
            return False

# Initialisation STANDARD
bot = USDCArbitrage(pairs=['BTC/USDC'])
risk = RiskExtension(bot)

# Test IMMÉDIAT
print("Validation BTC/USDC:", risk.validate('BTC/USDC'))
