import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import pandas as pd
import numpy as np
import json
import aiohttp
from datetime import datetime

@pytest.fixture(autouse=True)
def mock_env_vars():
    env_vars = {
        'TELEGRAM_TOKEN': 'test_token',
        'TELEGRAM_CHAT_ID': 'test_chat_id',
        'PERFORMANCE_LOG_DIR': 'logs/performance/',
        'BINANCE_API_KEY': 'test_binance_key',
        'BINANCE_API_SECRET': 'test_binance_secret',
        'BINGX_API_KEY': 'test_bingx_key',
        'BINGX_API_SECRET': 'test_bingx_secret'
    }
    with patch.dict(os.environ, env_vars):
        yield

@pytest.fixture
def mock_datetime():
    return datetime(2025, 5, 27, 6, 34, 21)

@pytest.fixture
def mock_json():
    def mock_dumps(obj, **kwargs):
        if isinstance(obj, (dict, list)):
            return json.dumps(obj)
        return str(obj)
    with patch('json.dumps', mock_dumps):
        yield

@pytest.fixture(autouse=True)
def mock_aiohttp():
    class MockAiohttp:
        def __init__(self, *args, **kwargs):
            pass
        
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc, tb):
            pass
            
        async def get(self, *args, **kwargs):
            return MagicMock(
                json=AsyncMock(return_value={
                    'symbol': 'BTCUSDT',
                    'price': '30000.00'
                })
            )
            
        async def post(self, *args, **kwargs):
            return MagicMock(
                json=AsyncMock(return_value={
                    'symbol': 'BTCUSDT',
                    'price': '30000.00'
                })
            )
    
    with patch('aiohttp.ClientSession', return_value=MockAiohttp()):
        yield
