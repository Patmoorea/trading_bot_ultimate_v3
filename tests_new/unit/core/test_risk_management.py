import pytest
import numpy as np
from tests_new.base_test import BaseTest

class TestRiskManagement(BaseTest):
    def test_risk_calculation(self):
        """Test risk calculation"""
        test_data = self.get_test_data(100)
        assert len(test_data) == 100
        assert isinstance(test_data, np.ndarray)

    def test_position_sizing(self):
        """Test position sizing"""
        test_data = self.get_test_data(100)
        assert np.all(test_data >= 0) and np.all(test_data <= 1)
