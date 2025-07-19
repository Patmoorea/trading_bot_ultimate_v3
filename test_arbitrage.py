import os
import asyncio
from dotenv import load_dotenv
import sys
import os
sys.path.insert(0, os.path.abspath('.')) 
# Adapte l'import selon ton projet (core/arbitrage_engine.py)
from src.strategies.arbitrage.multi_exchange.core.arbitrage_engine import ArbitrageEngine

# Charge les clés depuis .env
load_dotenv()

API_KEYS = {
    "binance": {
        "apiKey": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_API_SECRET", ""),
    },
    "gateio": {
        "apiKey": os.getenv("GATE_IO_API_KEY", ""),
        "secret": os.getenv("GATE_IO_API_SECRET", ""),
    },
    "okx": {
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_API_SECRET", ""),
        "password": os.getenv("OKX_API_PASSWORD", ""),
    },
    "bingx": {
        "apiKey": os.getenv("BINGX_API_KEY", ""),
        "secret": os.getenv("BINGX_API_SECRET", ""),
    },
    "blofin": {
        "apiKey": os.getenv("BLOFIN_API_KEY", ""),
        "secret": os.getenv("BLOFIN_API_SECRET", ""),
    },
}

async def main():
    # Instancie le moteur d'arbitrage
    engine = ArbitrageEngine(api_keys=API_KEYS)  # Adapte la signature si besoin

    print("Recherche d'opportunités d'arbitrage multi-exchange...")
    opportunities = await engine.check_arbitrage_async()  # Adapte le nom si besoin
    if not opportunities:
        print("Aucune opportunité trouvée.")
        return
    for opp in opportunities:
        print(f"Opportunité : {opp}")
        # Teste l'exécution simulée/réelle selon la config
        result = await engine.execute_arbitrage_async(opp)
        print(f"Résultat exécution : {result}")

if __name__ == "__main__":
    asyncio.run(main())
