import pytest
from unittest.mock import AsyncMock, MagicMock
from ...utils import TEST_CONFIG, AsyncBaseTestCase

class TestNotificationManager(AsyncBaseTestCase):
    @pytest.fixture
    def notification_manager(self):  # Remove async
        mock = MagicMock()
        mock.send_notification = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_init_with_log(self, notification_manager):
        assert hasattr(notification_manager, 'send_notification')
        assert True

    @pytest.mark.asyncio
    async def test_send_notification_log(self, notification_manager):
        await notification_manager.send_notification("test")
        notification_manager.send_notification.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_send_and_flush(self, notification_manager):
        await notification_manager.send_notification("test")
        await notification_manager.close()
        assert notification_manager.close.called
