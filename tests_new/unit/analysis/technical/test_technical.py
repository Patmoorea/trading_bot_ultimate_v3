from ....utils import BaseTestCase

class TestTechnical(BaseTestCase):
    def test_technical_analysis(self):
        data = self.get_test_data(size=100)
        assert len(data) >= 100
        assert all(col in data.columns for col in ['close', 'high', 'low'])

    def test_technical_signals(self):
        data = self.get_test_data(size=200)
        assert len(data) >= 200
        assert all(col in data.columns for col in ['close', 'volume'])
