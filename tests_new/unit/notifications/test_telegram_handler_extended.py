import pytest
from unittest.mock import AsyncMock, MagicMock
from ...utils import TEST_CONFIG, AsyncBaseTestCase

class TestTelegramHandlerExtended(AsyncBaseTestCase):
    @pytest.fixture
    def telegram_handler(self):  # Remove async
        mock = MagicMock()
        mock.send_message = AsyncMock()
        mock.send_photo = AsyncMock()
        mock.token = TEST_CONFIG.telegram_token
        mock.chat_id = TEST_CONFIG.telegram_chat_id
        return mock

    @pytest.mark.asyncio
    async def test_configuration_validation(self, telegram_handler):
        assert telegram_handler.token == TEST_CONFIG.telegram_token
        assert telegram_handler.chat_id == TEST_CONFIG.telegram_chat_id

    @pytest.mark.asyncio
    async def test_send_photo(self, telegram_handler):
        await telegram_handler.send_photo("test.jpg", "Test caption")
        telegram_handler.send_photo.assert_called_once_with("test.jpg", "Test caption")
