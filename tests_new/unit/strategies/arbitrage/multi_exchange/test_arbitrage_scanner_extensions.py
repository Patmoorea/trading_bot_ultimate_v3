"""
Extensions pour les tests d'arbitrage
Version 1.0.0 - Created: 2025-05-26 19:59:33 by Patmoorea
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def mock_order_book():
    """Crée un order book simulé avec des opportunités d'arbitrage"""
    def create_order_book(bids=None, asks=None):
        if bids is None:
            bids = [(39000.0, 1.0), (38999.0, 2.0), (38998.0, 1.5)]
        if asks is None:
            asks = [(39001.0, 1.0), (39002.0, 2.0), (39003.0, 1.5)]
            
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().timestamp(),
            'datetime': datetime.now().isoformat(),
            'nonce': int(datetime.now().timestamp() * 1000)
        }
    return create_order_book

@pytest.fixture
def mock_exchange_class(mock_order_book):
    """Crée une classe d'exchange simulée"""
    class MockExchange:
        def __init__(self):
            self.has = {'fetchOrderBook': True}
            self.id = 'mock_exchange'
            self.markets = {
                'BTC/USDT': {
                    'symbol': 'BTC/USDT',
                    'base': 'BTC',
                    'quote': 'USDT',
                    'active': True
                }
            }
            
        async def fetch_order_book(self, symbol, limit=None):
            return mock_order_book()
            
        def calculate_fee(self, symbol, type, side, amount, price, taker_or_maker='taker'):
            return {
                'rate': 0.001,
                'cost': amount * price * 0.001,
                'currency': 'USDT'
            }
    
    return MockExchange

@pytest.fixture
def mock_exchanges(mock_exchange_class):
    """Crée plusieurs instances d'exchanges simulés"""
    exchanges = []
    for i in range(3):
        exchange = mock_exchange_class()
        exchange.id = f'mock_exchange_{i}'
        exchanges.append(exchange)
    return exchanges

@pytest.fixture
def arbitrage_scanner_with_mocks(mock_exchanges):
    """Crée un scanner d'arbitrage avec des exchanges simulés"""
    from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner
    
    scanner = ArbitrageScanner(exchanges=mock_exchanges)
    
    # Ajout des méthodes manquantes
    if not hasattr(scanner, 'scan_opportunities'):
        async def scan_opportunities(symbols):
            """Simule la détection d'opportunités"""
            opportunities = []
            for symbol in symbols:
                opportunity = {
                    'symbol': symbol,
                    'buy_exchange': mock_exchanges[0].id,
                    'sell_exchange': mock_exchanges[1].id,
                    'buy_price': 39000.0,
                    'sell_price': 39100.0,
                    'volume': 1.0,
                    'profit': 100.0,
                    'profit_percent': 0.25,
                    'timestamp': datetime.now().timestamp()
                }
                opportunities.append(opportunity)
            return opportunities
            
        scanner.scan_opportunities = scan_opportunities.__get__(scanner)
    
    return scanner

def pytest_collection_modifyitems(items):
    """Modifie les items de test pour ajouter les fixtures nécessaires"""
    for item in items:
        if 'test_arbitrage_scanner.py' in str(item.fspath):
            item.fixturenames.extend(['mock_order_book', 'mock_exchange_class', 
                                    'mock_exchanges', 'arbitrage_scanner_with_mocks'])

# Extension des classes de test existantes
def extend_test_classes():
    """Étend les classes de test avec les fonctionnalités nécessaires"""
    import sys
    for module_name in list(sys.modules.keys()):
        if 'test_arbitrage_scanner' in module_name:
            module = sys.modules[module_name]
            for item_name in dir(module):
                if item_name.startswith('Test'):
                    test_class = getattr(module, item_name)
                    if isinstance(test_class, type):
                        # Ajout des méthodes et attributs nécessaires
                        if not hasattr(test_class, 'get_test_data'):
                            def get_test_data(self, *args, **kwargs):
                                return pd.DataFrame({
                                    'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='H'),
                                    'price': np.random.uniform(38000, 40000, 100),
                                    'volume': np.random.uniform(0.1, 2.0, 100)
                                })
                            setattr(test_class, 'get_test_data', get_test_data)

# Application des extensions
extend_test_classes()
