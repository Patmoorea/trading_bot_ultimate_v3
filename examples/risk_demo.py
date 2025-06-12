"""
DEMO OFFICIELLE - Utilise uniquement votre code existant
"""
import sys
from pathlib import Path

# Configuration du path exact comme dans vos scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.arbitrage.real_arbitrage_2 import USDCArbitrage
from src.strategies.arbitrage.core.risk_extension import RiskExtension

# Initialisation standard
bot = USDCArbitrage(pairs=['BTC/USDC'])
risk = RiskExtension(bot)

# Workflow
for pair in bot.pairs:
    is_valid, _ = risk.validate(pair)
    print(f"{pair}: {'✅' if is_valid else '❌'}")
