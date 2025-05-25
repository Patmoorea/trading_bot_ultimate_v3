import pytest
from tests_new.base_test import BaseTest
import numpy as np

class TestAIFull(BaseTest):
    def test_ai_data_shape(self):
        """Test AI data shapes"""
        data = self.get_test_data(100)
        assert len(data) == 100
        assert isinstance(data, np.ndarray)

    @pytest.mark.skip(reason="Requires full AI model")
    def test_ai_prediction(self):
        """Test AI prediction"""
        pass
