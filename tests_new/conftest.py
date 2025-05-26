# Created: 2025-05-25 18:39:54 UTC
# Author: Patmoorea
# Project: trading_bot_ultimate

import pytest
import logging

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture(autouse=True)
def mock_websocket(monkeypatch):
    """Mock websocket connections for all tests"""
    async def mock_connect(*args, **kwargs):
        class MockWebSocket:
            async def __aenter__(self):
                return self
                
            async def __aexit__(self, *args):
                pass
                
            async def recv(self):
                return "{}"
                
        return MockWebSocket()
    
    monkeypatch.setattr("websockets.connect", mock_connect)
