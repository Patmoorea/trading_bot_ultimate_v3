import asyncio
from src.strategies.arbitrage import ArbitrageBot
from utils.logger import get_logger

logger = get_logger()

async def test_bot():
    bot = ArbitrageBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(test_bot())
    except KeyboardInterrupt:
        logger.info("Test completed")
