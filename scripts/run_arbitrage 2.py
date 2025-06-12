#!/usr/bin/env python3
import asyncio
from src.strategies.arbitrage import ArbitrageEngine, ArbitrageExecutor
from src.data.stream import OrderBookStream

async def main():
    # Initialisation
    stream = OrderBookStream(['BTC/USDT', 'ETH/USDT'])
    engine = ArbitrageEngine()
    executor = ArbitrageExecutor()
    
    # Boucle principale
    async for orderbooks in stream.listen():
        opportunities = engine.find_opportunities(orderbooks)
        
        for opp in opportunities:
            try:
                result = executor.execute_trade(opp)
                print(f"Arbitrage exécuté: {result}")
            except Exception as e:
                print(f"Échec: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())
