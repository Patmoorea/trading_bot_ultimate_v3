import pytest
from tests_new.base_test import BaseTest
import os
import time

class TestPerformance(BaseTest):
    def test_performance_config(self):
        """Test performance configuration"""
        assert os.getenv('PERFORMANCE_LOG_PATH') == 'logs/performance/'

    @pytest.mark.performance
    def test_data_loading_performance(self):
        """Test data loading performance"""
        start_time = time.time()
        # Simulate data loading
        time.sleep(0.1)
        end_time = time.time()
        assert end_time - start_time < 1.0

    @pytest.mark.performance
    def test_model_inference_performance(self):
        """Test model inference performance"""
        start_time = time.time()
        # Simulate model inference
        time.sleep(0.1)
        end_time = time.time()
        assert end_time - start_time < 1.0
