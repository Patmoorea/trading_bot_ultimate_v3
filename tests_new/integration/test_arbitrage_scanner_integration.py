"""
Integration tests for arbitrage scanner
Version 1.0.0 - Created: 2025-05-19 03:22:15 by Patmoorea
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
import pytz
from src.utils.datetime_utils import get_utc_now

# Mock Exchange class
class MockExchange:
    def __init__(self, name, price_bias=0):
        self.name = name
        self.price_bias = price_bias

    async def get_ticker(self, symbol):
        base_price = 30000 + self.price_bias
        return {
            'bid': base_price - 50,
            'ask': base_price + 50,
            'last': base_price,
            'volume': 1.5,
            'timestamp': datetime.now(pytz.UTC).isoformat()
        }

# Import after mock definition
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner

@pytest.fixture
def mock_exchanges():
    return [
        MockExchange("Exchange1", 0),
        MockExchange("Exchange2", 200),
        MockExchange("Exchange3", -100)
    ]

@pytest.fixture
def scanner(mock_exchanges):
    return ArbitrageScanner(mock_exchanges)

@pytest.mark.asyncio
class TestArbitrageScannerIntegration:
    async def test_full_scan_cycle(self, scanner):
        """Test complete scanning cycle"""
        opportunities = await scanner.scan_opportunities(['BTC/USDT'])
        assert isinstance(opportunities, list)
        if opportunities:
            opp = opportunities[0]
            required_keys = {'buy_exchange', 'sell_exchange', 'symbol', 
                           'buy_price', 'sell_price', 'profit_pct', 'timestamp'}
            assert all(key in opp for key in required_keys)

    async def test_multiple_exchanges(self, scanner):
        """Test multiple exchange scanning"""
        opportunities = await scanner.scan_opportunities(['BTC/USDT'])
        
        exchange_pairs = {
            (opp['buy_exchange'], opp['sell_exchange'])
            for opp in opportunities
        }
        
        # Should have opportunities between different exchanges
        assert len(exchange_pairs) > 0
        
        # No self-trading
        for buy_ex, sell_ex in exchange_pairs:
            assert buy_ex != sell_ex

    async def test_timestamp_consistency(self, scanner):
        """Test timestamp consistency"""
        with patch('src.utils.datetime_utils.get_utc_now') as mock_now:
            mock_now.return_value = pd.Timestamp('2025-05-19 03:22:15')
            
            opportunities = await scanner.scan_opportunities(['BTC/USDT'])
            if opportunities:
                timestamps = {opp['timestamp'] for opp in opportunities}
                assert len(timestamps) == 1

    async def test_error_resilience(self, scanner):
        """Test scanner resilience to errors"""
        failing_exchange = Mock()
        failing_exchange.name = "FailingExchange"
        failing_exchange.get_ticker = Mock(side_effect=Exception("Test error"))
        scanner.exchanges.append(failing_exchange)
        
        opportunities = await scanner.scan_opportunities(['BTC/USDT'])
        assert isinstance(opportunities, list)
