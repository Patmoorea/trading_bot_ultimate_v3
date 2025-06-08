"""
Integration tests for Telegram Handler
"""

import pytest
import asyncio
import time
from decimal import Decimal
import logging
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator, Dict, Any, List 
from src.notifications.telegram_handler import TelegramHandler

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestConstants:
    """Test constants and configuration"""
    ALLOWED_USERS = [123, 456]
    UNAUTHORIZED_USERS = [789, 999, 1001]
    DEFAULT_TIMEOUT = 5.0
    RATE_LIMIT_WAIT = 0.02  # Déplacé ici comme constante

def run_async(coro):
    """Helper to run coroutine in synchronous context."""
    return asyncio.get_event_loop().run_until_complete(coro)

class TestTelegramHandler:
    """Test suite for TelegramHandler"""

    def setup_method(self, method):
        """Set up test cases."""
        self.handler = TelegramHandler(
            bot_token="test_token",
            allowed_users=TestConstants.ALLOWED_USERS,
            queue_size=100
        )
        run_async(self.handler.start())

    def teardown_method(self, method):
        """Clean up after test cases."""
        if hasattr(self, 'handler') and self.handler._running:
            run_async(self.handler.stop())

    async def wait_for_queue(self, timeout: float = TestConstants.DEFAULT_TIMEOUT):
        """Wait for queue to be empty."""
        try:
            start_time = asyncio.get_running_loop().time()
            while not self.handler.signal_queue.empty():
                if asyncio.get_running_loop().time() - start_time > timeout:
                    raise TimeoutError(f"Queue wait timeout after {timeout}s")
                await asyncio.sleep(0.1)
                
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error waiting for queue: {e}")
            raise

    async def wait_for_rate_limit(self, user_id: int, wait_time: float = TestConstants.RATE_LIMIT_WAIT):  # Corrigé ici
        """Wait for rate limit to reset."""
        await asyncio.sleep(wait_time)
