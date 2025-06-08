from ....utils import BaseTestCase

class TestIndicatorsV2(BaseTestCase):
    def test_indicator_input(self):
        data = self.get_test_data(size=100)
        assert 'close' in data.columns

    def test_indicator_calculation(self):
        data = self.get_test_data(size=100)
        assert len(data) == 100
