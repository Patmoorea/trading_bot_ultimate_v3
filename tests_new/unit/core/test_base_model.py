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

class TestBaseModel:
    def get_test_data(self):
        return {"price": 100, "volume": 0.5}

    def test_model_initialization(self):
        data = self.get_test_data()
        assert isinstance(data, dict)
    def get_test_data(self):
        return {"price": 100, "volume": 0.5}
