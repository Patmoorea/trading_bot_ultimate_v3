import pytest
from ...utils import BaseTestCase, TEST_CONFIG

class TestPerformance(BaseTestCase):
    def test_performance_calculation(self):
        data = self.get_test_data(size=100)
        assert len(data) == 100
        assert 'price' in data.columns

    def test_performance_logging(self):
        assert TEST_CONFIG.performance_log_dir == 'logs/performance/'
