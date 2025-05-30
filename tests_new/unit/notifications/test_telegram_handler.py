import pytest
from unittest.mock import AsyncMock, MagicMock
from ...utils import TEST_CONFIG, AsyncBaseTestCase

class TestTelegramHandler(AsyncBaseTestCase):
    @pytest.fixture
    def telegram_handler(self):
        mock = MagicMock()
        mock.send_message = AsyncMock()
        mock.token = TEST_CONFIG.telegram_token
        mock.chat_id = TEST_CONFIG.telegram_chat_id
        return mock

    @pytest.mark.asyncio
    async def test_send_message(self, telegram_handler):
        await telegram_handler.send_message("test")
        telegram_handler.send_message.assert_called_once_with("test")

    def test_configuration(self, telegram_handler):
        assert telegram_handler.token == TEST_CONFIG.telegram_token
        assert telegram_handler.chat_id == TEST_CONFIG.telegram_chat_id
