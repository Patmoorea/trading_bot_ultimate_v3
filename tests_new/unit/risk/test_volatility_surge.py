import pytest
from datetime import datetime
from decimal import Decimal

class TestVolatilitySurge:
    @pytest.fixture
    def volatility_config(self):
        return {
            'base_volatility': Decimal('0.02'),  # 2% volatilité de base
            'surge_multiplier': Decimal('3.0'),
            'timestamp': datetime(2025, 5, 25, 0, 48, 17)
        }

    def test_volatility_baseline(self, volatility_config):
        current_vol = Decimal('0.018')
        assert current_vol < volatility_config['base_volatility']

# EVOLUTION v1 - 2025-05-25 00:48:17
class TestVolatilitySurgeV1(TestVolatilitySurge):
    def test_volatility_spike(self, volatility_config):
        spike_level = Decimal('0.065')
        max_acceptable = volatility_config['base_volatility'] * volatility_config['surge_multiplier']
        assert spike_level > max_acceptable

# EVOLUTION v2 - 2025-05-25 00:49:15
class TestVolatilitySurgeV2(TestVolatilitySurgeV1):
    def test_volatility_clustering(self, volatility_config):
        cluster_count = 3
        max_clusters = 5
        assert cluster_count < max_clusters
