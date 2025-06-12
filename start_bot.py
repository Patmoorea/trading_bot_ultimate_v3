#!/usr/bin/env python
"""
Trading Bot Ultimate - Main Entry Point
Updated: 2025-05-27 16:32:27 UTC
Author: Patmoorea
"""

import asyncio
import logging
from datetime import datetime
from src.monitoring.dashboard import Dashboard
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner
from src.notifications import NotificationManager
from src.risk_management.position import PositionManager

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.start_time = datetime(2025, 5, 27, 16, 32, 27)
        self.dashboard = Dashboard()
        self.arbitrage_scanner = ArbitrageScanner()
        self.notification_manager = NotificationManager()
        self.position_manager = PositionManager()
        
    async def start(self):
        logger.info(f"Starting Trading Bot at {self.start_time}")
        
        # Démarrage des composants
        await asyncio.gather(
            self.dashboard.start(),
            self.arbitrage_scanner.start_scanning(),
            self.notification_manager.start(),
            self.position_manager.start_monitoring()
        )
        
    async def stop(self):
        logger.info("Stopping Trading Bot...")
        await asyncio.gather(
            self.dashboard.stop(),
            self.arbitrage_scanner.stop_scanning(),
            self.notification_manager.stop(),
            self.position_manager.stop_monitoring()
        )

async def main():
    bot = TradingBot()
    try:
        await bot.start()
        # Maintenir le bot en fonctionnement
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down bot...")
        await bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await bot.stop()
        raise

if __name__ == "__main__":
    asyncio.run(main())
