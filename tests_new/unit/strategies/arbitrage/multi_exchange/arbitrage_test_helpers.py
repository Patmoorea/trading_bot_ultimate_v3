"""
Helpers for Arbitrage Scanner Tests
Version 1.0.0 - Created: 2025-05-26 20:04:07 by Patmoorea
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict
import pytest
from unittest.mock import AsyncMock, MagicMock

class JSONSerializableScanner:
    """Wrapper pour rendre ArbitrageScanner sérialisable"""
    
    @staticmethod
    def default_json_encoder(obj: Any) -> Any:
        """Encodeur JSON personnalisé"""
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        if isinstance(obj, (datetime, AsyncMock, MagicMock)):
            return str(obj)
        return str(obj)

    @staticmethod
    def make_serializable(data: Any) -> Any:
        """Rend les données sérialisables"""
        if isinstance(data, dict):
            return {k: JSONSerializableScanner.make_serializable(v) for k, v in data.items()}
        if isinstance(data, list):
            return [JSONSerializableScanner.make_serializable(v) for v in data]
        if isinstance(data, (int, float, str, bool, type(None))):
            return data
        if isinstance(data, Decimal):
            return float(data)
        if hasattr(data, '__dict__'):
            return JSONSerializableScanner.make_serializable(data.__dict__)
        return str(data)

class ArbitrageScannerTestWrapper:
    """Wrapper pour les tests d'ArbitrageScanner"""
    
    def __init__(self, scanner):
        self._scanner = scanner
        self._setup_methods()

    def _setup_methods(self):
        """Configure les méthodes du scanner"""
        async def mock_fetch_order_book(symbol):
            return {
                'bids': [[39000.0, 1.0]],
                'asks': [[39100.0, 1.0]],
                'timestamp': datetime.now().timestamp(),
                'datetime': datetime.now().isoformat()
            }

        # Mock des méthodes nécessaires
        if not hasattr(self._scanner, 'fetch_order_book'):
            self._scanner.fetch_order_book = mock_fetch_order_book

    async def __call__(self, *args, **kwargs):
        """Permet d'utiliser le wrapper comme le scanner original"""
        result = await self._scanner(*args, **kwargs)
        return JSONSerializableScanner.make_serializable(result)

    def __getattr__(self, name):
        """Délègue les appels de méthode au scanner en les rendant sérialisables"""
        attr = getattr(self._scanner, name)
        if callable(attr):
            async def wrapped(*args, **kwargs):
                result = await attr(*args, **kwargs)
                return JSONSerializableScanner.make_serializable(result)
            return wrapped
        return JSONSerializableScanner.make_serializable(attr)

@pytest.fixture
def wrapped_scanner(scanner):
    """Fixture qui retourne un scanner enveloppé"""
    return ArbitrageScannerTestWrapper(scanner)

def make_test_data():
    """Crée des données de test cohérentes"""
    return {
        'exchanges': ['binance', 'kraken'],
        'symbols': ['BTC/USDT'],
        'orderbook': {
            'binance': {
                'BTC/USDT': {
                    'bids': [[39000.0, 1.0]],
                    'asks': [[39100.0, 1.0]]
                }
            },
            'kraken': {
                'BTC/USDT': {
                    'bids': [[38900.0, 1.0]],
                    'asks': [[39000.0, 1.0]]
                }
            }
        }
    }

# Patch pour pytest
def pytest_runtest_setup(item):
    """Setup pour chaque test"""
    if 'test_arbitrage_scanner.py' in str(item.fspath):
        if 'scanner' in item.fixturenames:
            item.fixturenames.append('wrapped_scanner')

# Application du patch aux tests existants
def patch_test_class(cls):
    """Applique les patches nécessaires à une classe de test"""
    original_setup = getattr(cls, 'setUp', None)
    
    def new_setup(self):
        """Setup étendu"""
        if original_setup:
            original_setup(self)
            
        # Ajout des attributs nécessaires
        self.test_data = make_test_data()
        self.json_encoder = JSONSerializableScanner.default_json_encoder

    cls.setUp = new_setup
    return cls

