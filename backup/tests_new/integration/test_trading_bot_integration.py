import pytest
from unittest.mock import patch, Mock
import asyncio
from datetime import datetime
from src.core.trading_bot import TradingBot

@pytest.fixture
def bot_config():
    return {
        'exchange_config': {
            'exchange_id': 'binance',
            'testnet': True
        },
        'telegram_config': {
            'token': 'test_token',
            'allowed_users': [123456789]
        },
        'trading_pairs': ['BTC/USDT', 'ETH/USDT'],
        'timeframes': ['1h', '4h']
    }

@pytest.fixture
def trading_bot(bot_config):
    return TradingBot(**bot_config)

@pytest.mark.asyncio
class TestTradingBotIntegration:
    async def test_start_stop(self, trading_bot):
        """Test bot start/stop"""
        # Start the bot
        task = asyncio.create_task(trading_bot.start())
        await asyncio.sleep(0.1)
        assert trading_bot._running
        
        # Stop the bot
        await trading_bot.stop()
        assert not trading_bot._running
        
        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_trading_cycle(self, trading_bot):
        """Test trading cycle execution"""
        await trading_bot._trading_cycle()
        # No exception should be raised

    async def test_pair_analysis(self, trading_bot):
        """Test pair analysis"""
        await trading_bot._analyze_pair('BTC/USDT')
        # No exception should be raised
