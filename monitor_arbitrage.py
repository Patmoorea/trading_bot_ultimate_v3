import asyncio
from modules.arbitrage_engine import ArbitrageEngine

class ArbitrageMonitor:
    def __init__(self):
        self.engine = ArbitrageEngine()
        self.stats = {
            'total_checks': 0,
            'opportunities_found': 0
        }
    
    async def run(self):
        while True:
            self.stats['total_checks'] += 1
            opportunities = await self.engine.check_opportunities_v2()
            if opportunities:
                self.stats['opportunities_found'] += len(opportunities)
                print(f"Stats: {self.stats}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    monitor = ArbitrageMonitor()
    asyncio.run(monitor.run())
    async def check_exchanges(self):
        available = await self.engine.get_available_exchanges()
        print(f"Exchanges actifs: {available}")
        return available

if __name__ == "__main__":
    async def main():
        monitor = ArbitrageMonitor()
        await monitor.check_exchanges()
        await monitor.run()
    
    asyncio.run(main())
