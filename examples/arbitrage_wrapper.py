import sys
from pathlib import Path

# Configuration spécifique à VOTRE dépôt
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    # Import exact selon votre structure Git
    from strategies.arbitrage.real_arbitrage_2 import USDCArbitrage
    print("✅ Import réussi depuis real_arbitrage_2.py")
    
    # Initialisation avec vos paramètres réels
    bot = USDCArbitrage(
        pairs=['BTC/USDC', 'ETH/USDC'],
        exchange_name='binance'
    )
    
    # Interface simplifiée
    def scan_opportunities():
        print("\n🔍 Scan des opportunités d'arbitrage...")
        try:
            opportunities = bot.scan_all_pairs()
            print(f"📊 Résultats : {len(opportunities)} opportunités")
            for pair, spread in opportunities:
                print(f"• {pair}: {spread*100:.2f}%")
        except Exception as e:
            print(f"❌ Erreur : {str(e)}")

    if __name__ == "__main__":
        scan_opportunities()

except ImportError as e:
    print(f"❌ Import impossible : {str(e)}")
    print("Vérifiez que :")
    print("1. Le fichier real_arbitrage_2.py existe bien")
    print("2. La classe USDCArbitrage est définie dedans")
