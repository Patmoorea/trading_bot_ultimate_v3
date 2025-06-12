import pytest
from unittest.mock import patch
import os

class TestTelegramIsolated:
    @pytest.fixture(autouse=True)
    def setup_env(self):
        original_token = os.environ.get('TELEGRAM_TOKEN')
        os.environ['TELEGRAM_TOKEN'] = 'test_token'
        yield
        if original_token:
            os.environ['TELEGRAM_TOKEN'] = original_token
        else:
            del os.environ['TELEGRAM_TOKEN']

    def test_telegram_env_isolation(self):
        assert os.environ.get('TELEGRAM_TOKEN') == 'test_token'
