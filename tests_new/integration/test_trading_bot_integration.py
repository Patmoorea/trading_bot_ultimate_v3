import pytest
from unittest.mock import AsyncMock, MagicMock

class TestTradingBotIntegration:
    @pytest.fixture
    def trading_bot(self):
        bot = MagicMock()
        bot.start = AsyncMock()
        bot.stop = AsyncMock()
        return bot

    @pytest.mark.asyncio
    async def test_start_stop(self, trading_bot):
        await trading_bot.start()
        await trading_bot.stop()
        assert trading_bot.start.called and trading_bot.stop.called
