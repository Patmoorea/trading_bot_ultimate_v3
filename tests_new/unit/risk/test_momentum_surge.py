import pytest
from datetime import datetime
from decimal import Decimal

class TestMomentumSurge:
    @pytest.fixture
    def momentum_params(self):
        return {
            'lookback_periods': 14,
            'surge_threshold': Decimal('3.0'),
            'current_time': datetime(2025, 5, 25, 0, 48, 17)
        }

    def test_momentum_calculation(self, momentum_params):
        roc = Decimal('2.5')  # Rate of Change
        assert roc < momentum_params['surge_threshold']

# EVOLUTION v1 - 2025-05-25 00:48:17
class TestMomentumSurgeV1(TestMomentumSurge):
    def test_acceleration(self, momentum_params):
        momentum_acceleration = Decimal('0.15')
        max_acceleration = Decimal('0.25')
        assert momentum_acceleration < max_acceleration

# EVOLUTION v2 - 2025-05-25 00:49:02
class TestMomentumSurgeV2(TestMomentumSurgeV1):
    def test_volume_confirmation(self, momentum_params):
        volume_surge = Decimal('2.8')
        min_confirmation = Decimal('2.0')
        assert volume_surge > min_confirmation
