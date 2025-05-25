import pytest
import numpy as np
from tests_new.base_test import BaseTest

class TestTechnical(BaseTest):
    def test_technical_analysis(self):
        """Test technical analysis functions"""
        test_data = self.get_test_data(100)
        assert len(test_data) == 100
        assert isinstance(test_data, np.ndarray)

    def test_technical_indicators(self):
        """Test technical indicators"""
        test_data = self.get_test_data(100)
        assert np.all(test_data >= 0) and np.all(test_data <= 1)
