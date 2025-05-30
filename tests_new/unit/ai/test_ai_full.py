import pytest
from ...utils import BaseTestCase

class TestAIFull(BaseTestCase):
    def test_ai_data_shape(self):
        data = self.get_test_data(size=100)
        assert len(data) == 100
        assert all(col in data.columns for col in ['timestamp', 'price', 'volume'])
