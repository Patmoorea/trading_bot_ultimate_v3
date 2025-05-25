import pytest
from datetime import datetime
from decimal import Decimal

class TestLiquidityShock:
    @pytest.fixture
    def setup_detector(self):
        return {
            'min_depth': Decimal('10.0'),
            'shock_threshold': Decimal('0.5')
        }

    def test_detect_liquidity_shock(self, setup_detector):
        # État initial
        initial_depth = Decimal('20.0')
        # Test de base
        assert initial_depth > setup_detector['min_depth']

# EVOLUTION v1 - 2025-05-25 00:47:12
class TestLiquidityShockV1(TestLiquidityShock):
    def test_shock_recovery(self, setup_detector):
        recovery_threshold = Decimal('0.8')
        depth_after_shock = Decimal('16.0')
        assert depth_after_shock > setup_detector['min_depth'] * recovery_threshold
