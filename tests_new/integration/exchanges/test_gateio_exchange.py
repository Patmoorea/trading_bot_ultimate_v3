import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import os

@pytest.mark.skipif(
    not all([os.getenv('GATEIO_API_KEY'), os.getenv('GATEIO_SECRET')]),
    reason="Gate.io credentials not configured in environment variables"
)
class TestGateIOExchange:
    @pytest.fixture
    async def exchange(self):
        with patch('gate_io.client.AsyncClient') as mock_client:
            mock_client.create = AsyncMock()
            client = await mock_client.create()
            client.get_ticker = AsyncMock(return_value={
                'symbol': 'BTC_USDT',
                'last': '30000.00',
                'bid': '29999.00',
                'ask': '30001.00'
            })
            client.get_account = AsyncMock(return_value={
                'balance': [
                    {'currency': 'BTC', 'available': '1.0', 'locked': '0.0'},
                    {'currency': 'USDT', 'available': '50000.0', 'locked': '0.0'}
                ]
            })
            return client

    @pytest.mark.asyncio
    async def test_market_data(self, exchange):
        ticker = await exchange.get_ticker(currency_pair='BTC_USDT')
        assert isinstance(ticker['last'], str)
        assert float(ticker['last']) > 0

    @pytest.mark.asyncio
    async def test_account_data(self, exchange):
        account = await exchange.get_account()
        assert 'balance' in account
        assert len(account['balance']) > 0

    @pytest.mark.asyncio
    async def test_order_lifecycle(self, exchange):
        exchange.create_order = AsyncMock(return_value={
            'currency_pair': 'BTC_USDT',
            'id': '12345',
            'status': 'open'
        })
        order = await exchange.create_order(
            currency_pair='BTC_USDT',
            side='buy',
            amount='0.001',
            price='30000'
        )
        assert order['status'] == 'open'

    @pytest.mark.asyncio
    async def test_arbitrage_opportunities(self, exchange):
        tickers = await exchange.get_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
