"""
Exchange Mocks for Testing
Version: 2.0.0 - Created: 2025-05-22 18:27:45 by Patmoorea
"""

from typing import Dict, Any
import asyncio
from datetime import datetime

class MockExchange:
    """Mock d'un exchange pour les tests"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orderbooks = {
            'BTC/USDC': {
                'bids': [[50000, 1]],
                'asks': [[50100, 1]]
            },
            'ETH/USDC': {
                'bids': [[3000, 10]],
                'asks': [[3010, 10]]
            },
            'ETH/BTC': {
                'bids': [[0.06, 5]],
                'asks': [[0.0601, 5]]
            }
        }
        
    async def fetch_order_book(self, symbol: str) -> Dict:
        """Simule la récupération d'un order book"""
        await asyncio.sleep(0.01)  # Petit délai simulé
        return self.orderbooks.get(symbol, {
            'bids': [[0, 0]],
            'asks': [[0, 0]]
        })
        
    async def close(self):
        """Simule la fermeture de la connexion"""
        pass

def get_mock_exchange(name: str) -> MockExchange:
    """Crée un mock d'exchange"""
    config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    }
    return MockExchange(config)
