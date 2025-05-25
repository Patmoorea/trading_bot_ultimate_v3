import pytest
import os
import numpy as np
from tests_new.base_test import BaseTest

class TestPerformance(BaseTest):
    def test_performance_calculation(self):
        """Test performance metrics calculation"""
        test_data = self.get_test_data(100)
        assert len(test_data) == 100
        assert isinstance(test_data, np.ndarray)

    def test_performance_logging(self):
        """Test performance logging"""
        assert os.getenv('PERFORMANCE_LOG_PATH') == 'logs/performance/'
        log_path = os.getenv('PERFORMANCE_LOG_PATH', 'logs/performance/')
        os.makedirs(log_path, exist_ok=True)
        assert os.path.exists(log_path)
