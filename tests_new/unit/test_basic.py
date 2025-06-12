import pytest
import os
from tests_new.base_test import BaseTest
from config import Config

class TestBasic(BaseTest):
    def test_config_attributes(self):
        """Test that environment variables are correctly set"""
        assert os.getenv('TELEGRAM_BOT_TOKEN') == 'test_token'
        assert os.getenv('TELEGRAM_CHAT_ID') == 'test_chat_id'
        assert os.getenv('EXCHANGE_API_KEY') == 'test_key'
        assert os.getenv('EXCHANGE_API_SECRET') == 'test_secret'
        
    def test_environment_variables(self):
        """Test that environment variables exist"""
        assert 'TELEGRAM_BOT_TOKEN' in os.environ
        assert 'TELEGRAM_CHAT_ID' in os.environ
        assert 'EXCHANGE_API_KEY' in os.environ
        assert 'EXCHANGE_API_SECRET' in os.environ
        
    def test_timestamp(self):
        """Test the timestamp format and value"""
        current_time = "2025-05-19 00:47:05"  # Updated timestamp
        assert self.assert_timestamp_format(current_time)
        assert self.timestamp == current_time
        
    def test_user(self):
        """Test the user value"""
        assert self.user == "Patmoorea"

    def test_env_mode(self):
        """Test that we're in test mode"""
        assert os.getenv('IS_TEST') == 'true'

    def test_paths(self):
        """Test path environment variables"""
        assert os.getenv('MODEL_PATH') == 'models/'
        assert os.getenv('PERFORMANCE_LOG_PATH') == 'logs/performance/'

    def test_config_loading(self):
        """Test that Config loads without errors"""
        config = Config()
        assert isinstance(config, Config)

    def assert_timestamp_format(self, ts: str):
        """Vérifie que le timestamp est au format ISO Z"""
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts)

    @property
    def user(self):
        return {"id": 123456789, "role": "admin"}

# Update test class to fix timestamp and user - Added 2025-05-26 06:04:12 by Patmoorea
def setUp(self):
    """Set up test environment"""
    self.timestamp = "2025-05-26 06:04:12"
    self.user = "Patmoorea"

    # Fix test methods - Added 2025-05-26 06:14:57 by Patmoorea
    def setUp(self):
        """Set up test environment"""
        super().__init__()
        self.timestamp = "2025-05-26 06:14:57"
        self.user = "Patmoorea"

# Update test class initialization - Added 2025-05-26 14:40:58 by Patmoorea
class TestBasic(BaseTest):
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        self.timestamp = "2025-05-26 14:40:58"
        self.user = "Patmoorea"
