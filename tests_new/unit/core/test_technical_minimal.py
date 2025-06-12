from ...utils import BaseTestCase

class TestTechnicalMinimal(BaseTestCase):
    def test_minimal_analysis(self):
        data = self.get_test_data(size=50)
        assert len(data) == 50

    def test_minimal_indicators(self):
        data = self.get_test_data(size=50)
        assert 'close' in data.columns
