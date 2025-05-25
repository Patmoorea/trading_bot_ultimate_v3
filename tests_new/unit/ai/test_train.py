import pytest
from tests_new.base_test import BaseTest
import numpy as np

class TestTrain(BaseTest):
    def test_training_data(self):
        """Test training data preparation"""
        data = self.get_test_data(100)
        assert len(data) == 100

    @pytest.mark.skip(reason="Requires training pipeline")
    def test_training_pipeline(self):
        """Test training pipeline"""
        pass
