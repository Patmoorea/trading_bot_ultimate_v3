import pytest
from unittest.mock import patch
import os

class TestTelegram:
    @pytest.fixture
    def telegram_config(self):
        with patch.dict(os.environ, {'TELEGRAM_TOKEN': 'test_token'}):
            yield

    def test_telegram_config(self, telegram_config):
        assert os.environ.get('TELEGRAM_TOKEN') == 'test_token'
