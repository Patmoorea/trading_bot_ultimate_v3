import pytest
from tests_new.base_test import BaseTest
import numpy as np

class TestTechnical(BaseTest):
    def test_technical_analysis(self):
        """Test technical analysis functions"""
        data = self.get_test_data(100)
        assert len(data) == 100

    def test_technical_signals(self):
        """Test technical signals generation"""
        data = self.get_test_data(100)
        assert len(data) > 0
