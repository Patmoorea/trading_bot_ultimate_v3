import sys
import os
from pathlib import Path

# Configuration du path spécifique à VOTRE projet
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import selon VOTRE structure
from src.strategies.arbitrage.multi_exchange import USDCArbitrage  # Adapté à votre chemin réel

def main():
    print("Initialisation du bot avec vos modules existants...")
    
    # Utilisation de VOTRE module exact
    arbitrage = USDCArbitrage(pairs=['BTC/USDC', 'ETH/USDC'])
    print(f"Module chargé : {arbitrage.__class__.__name__}")
    
    # Simulation d'usage
    try:
        opportunities = arbitrage.scan_all_pairs()
        print(f"Opportunités trouvées : {len(opportunities)}")
    except Exception as e:
        print(f"Erreur lors du scan : {str(e)}")

if __name__ == "__main__":
    main()
