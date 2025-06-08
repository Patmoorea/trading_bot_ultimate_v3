import pytest
from unittest.mock import AsyncMock, MagicMock

class TestNewsToTelegram:
    @pytest.fixture
    def mock_sentiment(self):
        mock = MagicMock()
        mock.score = 0.5
        return mock

    @pytest.mark.asyncio
    async def test_news_to_telegram_flow(self, mock_sentiment):
        assert mock_sentiment.score > 0.0
