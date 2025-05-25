import pytest
import os
import numpy as np
from tests_new.base_test import BaseTest

class TestBaseModel(BaseTest):
    def test_model_initialization(self):
        """Test base model initialization"""
        test_data = self.get_test_data(100)
        assert len(test_data) == 100
        assert isinstance(test_data, np.ndarray)

    def test_model_parameters(self):
        """Test base model parameters"""
        assert os.getenv('MODEL_PATH') == 'models/'
