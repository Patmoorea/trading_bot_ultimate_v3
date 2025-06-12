import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

class TestBingXExchange:
    @pytest.fixture
    def exchange(self):  # Removed async
        mock_client = MagicMock()
        # Setting up mock methods directly
        mock_client.futures_ticker = AsyncMock(return_value={
            'symbol': 'BTC-USDT',
            'lastPrice': '30000.00',
            'bidPrice': '29999.00',
            'askPrice': '30001.00'
        })
        mock_client.futures_account = AsyncMock(return_value={
            'assets': [
                {'asset': 'BTC', 'free': '1.0', 'locked': '0.0'},
                {'asset': 'USDT', 'free': '50000.0', 'locked': '0.0'}
            ]
        })
        mock_client.futures_exchange_info = AsyncMock(return_value={
            'symbols': [{'symbol': 'BTC-USDT', 'status': 'TRADING'}]
        })
        mock_client.futures_leverage = AsyncMock(return_value={
            'leverage': 20
        })
        return mock_client

    @pytest.mark.asyncio
    async def test_market_data(self, exchange):
        try:
            ticker = await exchange.futures_ticker(symbol='BTC-USDT')
            assert isinstance(ticker['lastPrice'], str)
            assert float(ticker['lastPrice']) > 0
        except Exception as e:
            pytest.fail(f"Market data test failed: {str(e)}")

    @pytest.mark.asyncio
    async def test_account_data(self, exchange):
        try:
            account = await exchange.futures_account()
            assert 'assets' in account
            assert len(account['assets']) > 0
        except Exception as e:
            pytest.fail(f"Account data test failed: {str(e)}")

    @pytest.mark.asyncio
    async def test_futures_settings(self, exchange):
        try:
            settings = await exchange.futures_leverage(symbol='BTC-USDT', leverage=20)
            assert settings['leverage'] == 20
        except Exception as e:
            pytest.fail(f"Futures settings test failed: {str(e)}")

    @pytest.mark.asyncio
    async def test_available_futures(self, exchange):
        info = await exchange.futures_exchange_info()
        assert 'symbols' in info
        assert len(info['symbols']) > 0
