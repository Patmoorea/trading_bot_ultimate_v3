import pytest
from tests_new.base_test import BaseTest
import os

class TestTelegram(BaseTest):
    def test_telegram_config(self):
        """Test that Telegram configuration is properly set"""
        assert os.getenv('TELEGRAM_BOT_TOKEN') == 'test_token'
        assert os.getenv('TELEGRAM_CHAT_ID') == 'test_chat_id'

    def test_telegram_env_vars_present(self):
        """Test that all required Telegram environment variables exist"""
        required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        for var in required_vars:
            assert var in os.environ, f"Missing environment variable: {var}"
            assert os.environ[var], f"Empty environment variable: {var}"

    @pytest.mark.skip(reason="Requires actual Telegram connection")
    def test_telegram_connection(self):
        """Test actual Telegram connection (skipped in test mode)"""
        pass
