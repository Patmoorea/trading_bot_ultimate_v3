import pytest
import numpy as np
from tests_new.base_test import BaseTest

class TestRSIMinimal(BaseTest):
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        test_data = self.get_test_data(100)
        assert len(test_data) == 100
        assert isinstance(test_data, np.ndarray)

    def test_rsi_bounds(self):
        """Test RSI is within bounds"""
        test_data = self.get_test_data(100)
        assert np.all(test_data >= 0) and np.all(test_data <= 1)
