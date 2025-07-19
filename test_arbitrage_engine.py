import asyncio
from src.strategies.arbitrage.multi_exchange.core.arbitrage_engine import MultiExchangeArbitrage

def main():
    arbitrage = MultiExchangeArbitrage()
    print("Recherche d'opportunités multi-exchange…")
    results = arbitrage.check_arbitrage(base='BTC', quote1='USDC', quote2='USDT')
    if not results:
        print("Aucune opportunité détectée.")
    else:
        for opp in results:
            print(f"Opportunité: {opp}")

if __name__ == "__main__":
    main()
