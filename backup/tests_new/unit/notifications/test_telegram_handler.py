"""
Unit tests for telegram handler
Version 1.0.0 - Created: 2025-05-19 05:47:22 by Patmoorea
"""

import pytest
import asyncio
from decimal import Decimal
import logging

from src.notifications.telegram_handler import TelegramHandler
from tests_new.utils.test_base import AsyncBaseTestCase

class TestTelegramHandler(AsyncBaseTestCase):
    
    @pytest.fixture
    def handler(self):
        """Create telegram handler instance"""
        return TelegramHandler(
            bot_token="test_token",
            allowed_users=[123, 456],
            queue_size=10
        )

    @pytest.mark.asyncio
    async def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler.bot_token == "test_token"
        assert handler.allowed_users == {123, 456}
        assert handler.signal_queue.maxsize == 10
        assert not handler._running

    @pytest.mark.asyncio
    async def test_authorization(self, handler):
        """Test user authorization"""
        assert handler.is_authorized(123)
        assert handler.is_authorized(456)
        assert not handler.is_authorized(789)

    @pytest.mark.asyncio
    async def test_send_signal(self, handler):
        """Test signal sending"""
        await handler.start()
        try:
            signal = {
                "pair": "BTC/USDT",
                "action": "BUY",
                "price": Decimal("30000"),
                "amount": Decimal("0.1")
            }
            
            # Test authorized user
            assert await handler.send_signal(signal, 123)
            
            # Test unauthorized user
            assert not await handler.send_signal(signal, 789)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            assert handler.signal_queue.empty()
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_signal_queue(self, handler):
        """Test signal queuing"""
        await handler.start()
        try:
            signals = [
                {
                    "pair": f"BTC/USDT_{i}",
                    "action": "BUY",
                    "price": Decimal("30000"),
                    "amount": Decimal("0.1")
                }
                for i in range(5)
            ]
            
            # Send multiple signals
            for signal in signals:
                assert await handler.send_signal(signal, 123)
            
            # Give some time for processing
            await asyncio.sleep(1.0)
            
            # Queue should be empty after processing
            assert handler.signal_queue.empty()
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, handler):
        """Test rate limiting"""
        await handler.start()
        try:
            signal = {
                "pair": "BTC/USDT",
                "action": "BUY",
                "price": Decimal("30000"),
                "amount": Decimal("0.1")
            }
            
            # First signal should succeed
            assert await handler.send_signal(signal, 123)
            
            # Immediate second signal should fail (rate limited)
            assert not await handler.send_signal(signal, 123)
            
            # Wait for rate limit to expire
            await asyncio.sleep(1.0)
            
            # Third signal should succeed
            assert await handler.send_signal(signal, 123)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            assert handler.signal_queue.empty()
        finally:
            await handler.stop()

    @pytest.mark.asyncio
    async def test_stop_handler(self, handler):
        """Test handler stop"""
        await handler.start()
        assert handler._running
        await handler.stop()
        assert not handler._running
        assert handler._worker_task is None or handler._worker_task.done()
