import pytest
from datetime import datetime
from decimal import Decimal

class TestLiveEvolution:
    @pytest.fixture
    def market_state(self):
        return {
            'timestamp': datetime(2025, 5, 25, 0, 48, 17),
            'last_price': Decimal('19850.75'),
            'volume_24h': Decimal('12547.89')
        }

    def test_initial_state(self, market_state):
        assert isinstance(market_state['last_price'], Decimal)
        assert market_state['timestamp'].date().year == 2025

# EVOLUTION v1 - 2025-05-25 00:48:17
class TestLiveEvolutionV1(TestLiveEvolution):
    def test_volume_tracking(self, market_state):
        hourly_volume = Decimal('523.66')
        assert hourly_volume <= market_state['volume_24h']
        assert hourly_volume > Decimal('0')

# EVOLUTION v2 - 2025-05-25 00:48:45
class TestLiveEvolutionV2(TestLiveEvolutionV1):
    def test_price_velocity(self, market_state):
        velocity = Decimal('12.5')  # $/seconde
        max_velocity = Decimal('50.0')
        assert velocity < max_velocity
