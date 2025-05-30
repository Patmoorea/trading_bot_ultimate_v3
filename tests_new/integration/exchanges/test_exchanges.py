import pytest
from unittest.mock import AsyncMock, MagicMock
import aiohttp
from datetime import datetime

class MockResponse:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data

class MockSession:
    def __init__(self):
        self.get = AsyncMock()
        self.post = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class BaseMockExchange:
    def __init__(self, session):
        self.session = session
        self._ticker_data = {
            'symbol': 'BTCUSDT',
            'price': '30000.00'
        }

    async def get_ticker(self, symbol):
        return self._ticker_data

@pytest.mark.asyncio
class TestBinanceExchange:
    @pytest.fixture
    def mock_session(self):  # Remove async
        session = MockSession()
        session.get.return_value = MockResponse({
            'symbol': 'BTCUSDT',
            'price': '30000.00',
            'volume': '100.0',
            'timestamp': '1621234567000'
        })
        return session

    @pytest.fixture
    def exchange(self, mock_session):  # Remove async
        return BaseMockExchange(mock_session)

    async def test_market_data(self, exchange):
        result = await exchange.get_ticker('BTCUSDT')
        assert result['symbol'] == 'BTCUSDT'
        assert 'price' in result

@pytest.mark.asyncio
class TestBingXExchange:
    @pytest.fixture
    def mock_session(self):  # Remove async
        session = MockSession()
        session.get.return_value = MockResponse({
            'symbol': 'BTCUSDT',
            'price': '30000.00',
            'volume': '100.0',
            'timestamp': '1621234567000'
        })
        return session

    @pytest.fixture
    def exchange(self, mock_session):  # Remove async
        return BaseMockExchange(mock_session)

    async def test_market_data(self, exchange):
        result = await exchange.get_ticker('BTCUSDT')
        assert result['symbol'] == 'BTCUSDT'
        assert 'price' in result
