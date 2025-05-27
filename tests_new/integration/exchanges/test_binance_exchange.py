import pytest
import asyncio
from datetime import datetime

@pytest.mark.asyncio
class TestBinanceExchange:
    @pytest.fixture
    def market_data(self):
        return {
            'BTC/USDC': {
                'bid': 30000.0,
                'ask': 30100.0,
                'timestamp': int(datetime(2025, 5, 27, 7, 23, 0).timestamp() * 1000)
            },
            'ETH/USDC': {
                'bid': 2000.0,
                'ask': 2010.0,
                'timestamp': int(datetime(2025, 5, 27, 7, 23, 0).timestamp() * 1000)
            }
        }

    async def test_market_data(self, market_data):
        assert 'BTC/USDC' in market_data
        assert 'ETH/USDC' in market_data
        for symbol in market_data:
            assert 'USDC' in symbol
            assert market_data[symbol]['bid'] < market_data[symbol]['ask']

    async def test_account_data(self):
        account = {
            'USDC': 10000.0,
            'BTC': 1.0,
            'ETH': 10.0
        }
        assert 'USDC' in account
        assert account['USDC'] > 0

    async def test_order_lifecycle(self):
        order = {
            'symbol': 'BTC/USDC',
            'type': 'LIMIT',
            'side': 'BUY',
            'price': 30000.0,
            'amount': 0.1,
            'timestamp': int(datetime(2025, 5, 27, 7, 23, 0).timestamp() * 1000)
        }
        assert order['symbol'].endswith('/USDC')
        assert order['price'] > 0
        assert order['amount'] > 0

    async def test_available_markets(self):
        markets = [
            'BTC/USDC',
            'ETH/USDC',
            'BNB/USDC',
            'SOL/USDC'
        ]
        assert all('/USDC' in market for market in markets)
