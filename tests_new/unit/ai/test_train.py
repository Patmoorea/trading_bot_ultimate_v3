from ...utils import BaseTestCase

class TestTrain(BaseTestCase):
    def test_training_data(self):
        data = self.get_test_data(size=100)
        assert len(data) == 100
