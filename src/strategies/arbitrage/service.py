"""
Service d'arbitrage
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List
from datetime import datetime

class MoteurArbitrage:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.exchanges = {}
        self.min_spread = config.get('min_spread', 0.005)
        self.pairs = config.get('pairs', ['BTC/USDC', 'ETH/USDC'])
        self.test_mode = config.get('test_mode', False)
        
    async def initialize(self):
        try:
            if self.test_mode:
                from tests_new.mocks.exchange_mock import get_mock_exchange
                self.exchanges = {
                    'binance': get_mock_exchange('binance'),
                    'gateio': get_mock_exchange('gateio'),
                    'okx': get_mock_exchange('okx')
                }
            else:
                from ccxt.async_support import binance, gateio, okx
                self.exchanges = {
                    'binance': binance({'enableRateLimit': True}),
                    'gateio': gateio({'enableRateLimit': True}),
                    'okx': okx({'enableRateLimit': True})
                }
        except Exception as e:
            self.logger.error(f"Erreur initialisation exchanges: {str(e)}")
            raise
            
    async def fetch_orderbook(self, exchange, symbol: str) -> Dict:
        try:
            orderbook = await exchange.fetch_order_book(symbol)
            return {
                'bids': orderbook['bids'][:5],
                'asks': orderbook['asks'][:5],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Erreur fetch orderbook {symbol}: {str(e)}")
            return None
            
    async def close(self):
        for exchange in self.exchanges.values():
            await exchange.close()
