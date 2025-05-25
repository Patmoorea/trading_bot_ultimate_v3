"""
Unit tests for arbitrage scanner
Version 1.0.0 - Created: 2025-05-19 05:16:37 by Patmoorea
"""

import pytest
from decimal import Decimal
from unittest.mock import patch
import pandas as pd
from datetime import datetime
import pytz

from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner
from src.utils.datetime_utils import get_utc_now, format_timestamp

# Test timestamp that will be used consistently across all tests
TEST_TIMESTAMP = pd.Timestamp('2025-05-19 05:16:37+00:00').to_pydatetime().replace(tzinfo=pytz.UTC)

class MockDateTime:
    """Mock datetime for consistent testing"""
    @staticmethod
    def now(tz=None):
        return TEST_TIMESTAMP

class TestMockExchange:
    """Mock exchange for unit testing"""
    def __init__(self, name, price_bias=0):
        self.name = name.lower()
        self.price_bias = Decimal(str(price_bias))

    async def get_ticker(self, symbol):
        """Mock get_ticker with consistent timestamp"""
        base_price = Decimal('30000') + self.price_bias
        return {
            'bid': base_price - Decimal('50'),
            'ask': base_price + Decimal('50'),
            'last': base_price,
            'volume': Decimal('1.5'),
            'timestamp': format_timestamp(TEST_TIMESTAMP)
        }

@pytest.fixture(autouse=True)
def mock_datetime(monkeypatch):
    """Override datetime.now globally"""
    monkeypatch.setattr('src.utils.datetime_utils.datetime', MockDateTime)
    return MockDateTime

@pytest.fixture
def mock_exchanges():
    """Create mock exchanges"""
    exchange1 = TestMockExchange("binance")
    exchange2 = TestMockExchange("bybit", price_bias=200)
    return [exchange1, exchange2]

@pytest.fixture
def scanner(mock_exchanges):
    """Create scanner instance"""
    return ArbitrageScanner(exchanges=mock_exchanges,
                           min_profit_threshold=Decimal('0.001'),
                           max_price_deviation=Decimal('0.05'))

@pytest.mark.asyncio
class TestArbitrageScanner:
    def test_initialization(self, scanner, mock_exchanges):
        """Test scanner initialization"""
        assert len(scanner.exchanges) == len(mock_exchanges)
        assert scanner.min_profit_threshold == Decimal('0.001')
        assert scanner.max_price_deviation == Decimal('0.05')

    def test_validate_symbol(self, scanner):
        """Test symbol validation"""
        assert scanner._validate_symbol('BTC/USDC', 'binance') is True
        assert scanner._validate_symbol('ETH/USDC', 'binance') is True
        assert scanner._validate_symbol('BTC/USDT', 'binance') is False
        assert scanner._validate_symbol('BTC/USDT', 'bybit') is True
        assert scanner._validate_symbol('ETH/USDT', 'bybit') is True
        assert scanner._validate_symbol('BTC/USDC', 'bybit') is False
        assert scanner._validate_symbol('INVALID', 'binance') is False

    def test_get_equivalent_symbol(self, scanner):
        """Test symbol conversion"""
        assert scanner._get_equivalent_symbol('BTC/USDT', 'binance') == 'BTC/USDC'
        assert scanner._get_equivalent_symbol('BTC/USDC', 'bybit') == 'BTC/USDT'
        assert scanner._get_equivalent_symbol('BTC/USDC', 'binance') == 'BTC/USDC'
        assert scanner._get_equivalent_symbol('BTC/USDT', 'bybit') == 'BTC/USDT'

    def test_calculate_profit(self, scanner):
        """Test profit calculation"""
        buy_price = Decimal('30000')
        sell_price = Decimal('30300')
        expected_profit = (sell_price - buy_price) / buy_price
        profit = scanner._calculate_profit(buy_price, sell_price)
        assert profit == expected_profit
        assert profit > scanner.min_profit_threshold

    def test_check_price_deviation(self, scanner):
        """Test price deviation check"""
        valid_prices = [Decimal('30000'), Decimal('30100'), 
                       Decimal('30200'), Decimal('30300')]
        assert scanner._check_price_deviation(valid_prices)

        invalid_prices = [Decimal('30000'), Decimal('35000')]
        assert not scanner._check_price_deviation(invalid_prices)

    async def test_scan_opportunities(self, scanner):
        """Test opportunity scanning"""
        opportunities = await scanner.scan_opportunities(['BTC/USDT'])
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        if opportunities:
            opp = opportunities[0]
            required_keys = {'buy_exchange', 'sell_exchange', 'symbol', 
                           'buy_price', 'sell_price', 'profit_pct', 'timestamp'}
            assert all(key in opp for key in required_keys)
            assert opp['timestamp'] == format_timestamp(TEST_TIMESTAMP)

    def test_quote_currency_mapping(self, scanner):
        """Test quote currency mapping"""
        assert scanner._get_exchange_quote_currency('binance') == 'USDC'
        assert scanner._get_exchange_quote_currency('bybit') == 'USDT'
        assert scanner._get_exchange_quote_currency('unknown') == 'USDT'
