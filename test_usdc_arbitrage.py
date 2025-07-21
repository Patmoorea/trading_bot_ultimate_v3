import asyncio
import logging
from src.strategies.arbitrage.usdc_arbitrage import USDCArbitrage

config = {
    "min_spread": 0.002,
    "timeout": 30000,
    "exchanges": ["binance", "gateio"],
}

async def main():
    logging.basicConfig(level=logging.INFO)
    arb = USDCArbitrage(config)
    print("Scan USDC pairs...")
    opportunities = await arb.get_opportunities()
    print("Opportunités trouvées :")
    for pair, spread in opportunities:
        print(f"{pair}: spread={spread:.4%}")

if __name__ == "__main__":
    asyncio.run(main())