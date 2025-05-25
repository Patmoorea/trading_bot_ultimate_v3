"""
Integration tests for arbitrage scanner
Version 1.0.0 - Created: 2025-05-19 05:19:34 by Patmoorea
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))

import pytest
from decimal import Decimal
from unittest.mock import patch, Mock
import pandas as pd
from datetime import datetime
import pytz

from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner
from src.utils.datetime_utils import get_utc_now, format_timestamp

class TestMockExchange:
    """Enhanced mock exchange for testing"""
    def __init__(self, name, price_bias=0):
        self.name = name.lower()
        self.price_bias = Decimal(str(price_bias))
        self._last_ticker_timestamp = None

    async def get_ticker(self, symbol):
        """Mock get_ticker with consistent timestamp"""
        self._last_ticker_timestamp = pytest.TEST_TIMESTAMP
        base_price = Decimal('30000') + self.price_bias
        return {
            'bid': base_price - Decimal('50'),
            'ask': base_price + Decimal('50'),
            'last': base_price,
            'volume': Decimal('1.5'),
            'timestamp': format_timestamp(pytest.TEST_TIMESTAMP)
        }

@pytest.fixture
def mock_exchanges():
    """Create mock exchanges with different price biases"""
    return [
        TestMockExchange("binance", 0),      # Will use USDC
        TestMockExchange("bybit", 200),      # Will use USDT
        TestMockExchange("bitfinex", -100)   # Will use USDT
    ]

@pytest.fixture
def scanner(mock_exchanges):
    """Create scanner instance"""
    return ArbitrageScanner(mock_exchanges, 
                           min_profit_threshold=Decimal('0.001'), 
                           max_price_deviation=Decimal('0.05'))

@pytest.mark.asyncio
class TestArbitrageScannerIntegration:
    async def test_full_scan_cycle(self, scanner):
        """Test complete scanning cycle"""
        opportunities = await scanner.scan_opportunities(['BTC/USDT'])
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        if opportunities:
            opp = opportunities[0]
            required_keys = {'buy_exchange', 'sell_exchange', 'symbol', 
                           'buy_price', 'sell_price', 'profit_pct', 'timestamp'}
            assert all(key in opp for key in required_keys)
            assert opp['timestamp'] == format_timestamp(pytest.TEST_TIMESTAMP)

    async def test_multiple_exchanges(self, scanner):
        """Test multiple exchange scanning"""
        opportunities = await scanner.scan_opportunities(['BTC/USDT'])
        assert len(opportunities) > 0
        
        exchange_pairs = {
            (opp['buy_exchange'], opp['sell_exchange'])
            for opp in opportunities
        }
        
        assert len(exchange_pairs) > 0
        
        for buy_ex, sell_ex in exchange_pairs:
            assert buy_ex != sell_ex
            
        timestamps = {opp['timestamp'] for opp in opportunities}
        assert len(timestamps) == 1
        assert list(timestamps)[0] == format_timestamp(pytest.TEST_TIMESTAMP)

    async def test_timestamp_consistency(self, scanner):
        """Test timestamp consistency"""
        opportunities = await scanner.scan_opportunities(['BTC/USDT'])
        assert len(opportunities) > 0
        
        timestamps = {opp['timestamp'] for opp in opportunities}
        assert len(timestamps) == 1, f"Multiple timestamps found: {timestamps}"
        assert list(timestamps)[0] == format_timestamp(pytest.TEST_TIMESTAMP)

    async def test_cross_currency_arbitrage(self, scanner):
        """Test arbitrage between USDC and USDT pairs"""
        opportunities = await scanner.scan_opportunities(['BTC/USDT', 'BTC/USDC'])
        assert len(opportunities) > 0
        
        cross_currency_opportunities = [
            opp for opp in opportunities
            if (opp['buy_exchange'] == 'binance' or opp['sell_exchange'] == 'binance')
        ]
        
        assert len(cross_currency_opportunities) > 0
        
        timestamps = {opp['timestamp'] for opp in opportunities}
        assert len(timestamps) == 1
        assert list(timestamps)[0] == format_timestamp(pytest.TEST_TIMESTAMP)
