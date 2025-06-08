"""
WRAPPER OFFICIEL - Utilise VOTRE code exact
"""
import sys
from pathlib import Path

# Chemin ABSOLU vérifié
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import selon VOTRE implémentation réelle
from strategies.arbitrage.real_arbitrage_2 import USDCArbitrage

def main():
    print("Initialisation avec VOS paramètres exacts...")
    bot = USDCArbitrage(pairs=['BTC/USDC'], exchange_name='binance')
    
    print("Méthodes disponibles:", [m for m in dir(bot) if not m.startswith('_')])
    
    try:
        opportunities = bot.scan_all_pairs()
        print(f"Résultats du scan : {opportunities}")
    except Exception as e:
        print(f"Erreur contrôlée : {str(e)}")

if __name__ == "__main__":
    main()
