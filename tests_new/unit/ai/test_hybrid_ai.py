from ...utils import BaseTestCase

class TestHybridAI(BaseTestCase):
    def test_hybrid_model_input(self):
        data = self.get_test_data(size=100)
        assert len(data) == 100
