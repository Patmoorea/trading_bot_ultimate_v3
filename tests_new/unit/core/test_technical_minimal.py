import pytest
import numpy as np
from tests_new.base_test import BaseTest

class TestTechnicalMinimal(BaseTest):
    def test_minimal_analysis(self):
        """Test minimal technical analysis"""
        test_data = self.get_test_data(100)
        assert len(test_data) == 100
        assert isinstance(test_data, np.ndarray)

    def test_minimal_indicators(self):
        """Test minimal indicators"""
        test_data = self.get_test_data(100)
        assert np.all(test_data >= 0) and np.all(test_data <= 1)
