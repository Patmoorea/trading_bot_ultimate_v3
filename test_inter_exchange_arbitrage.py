import asyncio
import os
from dotenv import load_dotenv
from modules.inter_exchange_arbitrage import InterExchangeArbitrage

# Charge les clés API depuis le .env
load_dotenv()

# Binance
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Gate.io
GATE_IO_API_KEY = os.getenv('GATE_IO_API_KEY', '')
GATE_IO_API_SECRET = os.getenv('GATE_IO_API_SECRET', '')

# BingX
BINGX_API_KEY = os.getenv('BINGX_API_KEY', '')
BINGX_API_SECRET = os.getenv('BINGX_API_SECRET', '')

# Blofin
BLOFIN_API_KEY = os.getenv('BLOFIN_API_KEY', '')
BLOFIN_API_SECRET = os.getenv('BLOFIN_API_SECRET', '')

# OKX
OKX_API_KEY = os.getenv('OKX_API_KEY', '')
OKX_API_SECRET = os.getenv('OKX_API_SECRET', '')
OKX_API_PASSWORD = os.getenv('OKX_API_PASSWORD', '')

config = {
    'exchanges': ['binance', 'gateio', 'bingx', 'blofin', 'okx'],
    'binance_api_key': BINANCE_API_KEY,
    'binance_api_secret': BINANCE_API_SECRET,
    'gateio_api_key': GATE_IO_API_KEY,
    'gateio_api_secret': GATE_IO_API_SECRET,
    'bingx_api_key': BINGX_API_KEY,
    'bingx_api_secret': BINGX_API_SECRET,
    'blofin_api_key': BLOFIN_API_KEY,
    'blofin_api_secret': BLOFIN_API_SECRET,
    'okx_api_key': OKX_API_KEY,
    'okx_api_secret': OKX_API_SECRET,
    'okx_api_password': OKX_API_PASSWORD,
    'min_profit': 0.5,
    'symbols': ['BTC/USDT', 'ETH/USDT'],
    'fees': {
        'binance': 0.1,
        'gateio': 0.1,
        'bingx': 0.1,
        'blofin': 0.1,
        'okx': 0.1
    },
    'withdrawal_fees': {
        'binance': {'BTC': 0.0005, 'ETH': 0.005},
        'gateio': {'BTC': 0.0005, 'ETH': 0.005},
        'bingx': {'BTC': 0.0005, 'ETH': 0.005},
        'blofin': {'BTC': 0.0005, 'ETH': 0.005},
        'okx': {'BTC': 0.0005, 'ETH': 0.005}
    },
}

async def main():
    arb = InterExchangeArbitrage(config)
    print("Recherche d'opportunités d'arbitrage...")
    opportunities = await arb.find_opportunities()
    for opp in opportunities:
        print(f"Arbitrage possible: {opp}")
        # Teste l'exécution réelle (simulation si pas assez de fonds)
        result = await arb.execute_arbitrage(opp, amount=20)  # 20 USDT par exemple
        print(f"Résultat exécution: {result}")
    await arb.close()

if __name__ == "__main__":
    asyncio.run(main())
