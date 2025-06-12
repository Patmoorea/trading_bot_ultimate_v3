import pytest
from datetime import datetime
from decimal import Decimal

class TestMarketCrash:
    @pytest.fixture
    def crash_params(self):
        return {
            'threshold': Decimal('-0.10'),
            'timeframe': 300  # 5 minutes
        }

    def test_detect_crash(self, crash_params):
        initial_price = Decimal('20000')
        crash_price = Decimal('17000')
        assert (crash_price - initial_price) / initial_price < crash_params['threshold']

# EVOLUTION v1 - 2025-05-25 00:47:45
class TestMarketCrashV1(TestMarketCrash):
    def test_flash_crash(self, crash_params):
        flash_duration = 60  # 1 minute
        assert flash_duration < crash_params['timeframe']
