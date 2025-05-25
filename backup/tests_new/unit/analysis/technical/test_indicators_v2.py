import pytest
from tests_new.base_test import BaseTest
import numpy as np

class TestIndicatorsV2(BaseTest):
    def test_indicator_input(self):
        """Test indicator input validation"""
        data = self.get_test_data(100)
        assert len(data) == 100
        assert isinstance(data, np.ndarray)

    def test_indicator_calculation(self):
        """Test indicator calculation"""
        data = self.get_test_data(100)
        # Add your indicator calculation tests here
        assert len(data) > 0
