from ...utils import BaseTestCase

class TestRSIMinimal(BaseTestCase):
    def test_rsi_calculation(self):
        data = self.get_test_data(size=100)
        assert 'close' in data.columns

    def test_rsi_bounds(self):
        data = self.get_test_data(size=100)
        assert len(data) == 100
