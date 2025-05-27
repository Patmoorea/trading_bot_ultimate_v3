import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime

class MockResponse:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data

@pytest.fixture
def mock_datetime():
    return datetime(2025, 5, 27, 6, 58, 26)

@pytest.fixture
def setup_json_serialization():
    original_dumps = json.dumps
    def json_dumps(obj, *args, **kwargs):
        if isinstance(obj, (dict, list)):
            return original_dumps(obj, *args, **kwargs)
        return str(obj)
    
    with patch('json.dumps', side_effect=json_dumps) as mock:
        yield mock

@pytest.fixture
def mock_exchange_response():
    return {
        'symbol': 'BTCUSDT',
        'price': '30000.00',
        'volume': '100.0',
        'timestamp': '1621234567000'
    }

@pytest.fixture
def mock_session(mock_exchange_response):
    class MockClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, *args, **kwargs):
            return MockResponse(mock_exchange_response)

        async def post(self, *args, **kwargs):
            return MockResponse(mock_exchange_response)

    return MockClientSession()

@pytest.fixture(autouse=True)
def mock_aiohttp_session(mock_session):
    with patch('aiohttp.ClientSession', return_value=mock_session):
        yield
