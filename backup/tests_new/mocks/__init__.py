from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

# Mock pour transformers.pipeline
class MockPipeline:
    def __call__(self, text):
        return [{'label': 'positive', 'score': 0.85}]

# Mock pour TextBlob
class MockTextBlob:
    class Sentiment:
        def __init__(self):
            self.polarity = 0.5
            self.subjectivity = 0.5

    def __init__(self, text):
        self.sentiment = self.Sentiment()

# Mock pour Telegram
class MockBot:
    def __init__(self, *args, **kwargs):
        self.send_message = MagicMock(return_value=True)

class MockUpdate:
    def __init__(self):
        self.effective_user = Mock()
        self.effective_user.id = 123456789
        self.message = Mock()
        self.message.reply_text = MagicMock()

class MockApplication:
    @staticmethod
    def builder():
        return MockApplicationBuilder()

class MockApplicationBuilder:
    def token(self, *args):
        return self

    def build(self):
        app = Mock()
        app.add_handler = MagicMock()
        app.run_polling = MagicMock()
        app.stop = MagicMock()
        return app

# Mock pour Exchange
class MockExchange:
    def __init__(self):
        self.fetch_ohlcv = MagicMock(return_value=[
            [1609459200000, 35000, 35100, 34900, 35000, 1000]
            for _ in range(100)
        ])
        self.get_ticker = MagicMock(return_value={'last': 35000})
        self.fetch_order = MagicMock()
        self.create_order = MagicMock()
        self.cancel_order = MagicMock()

# Données sample
def get_sample_ohlcv():
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='1h'),
        'open': np.random.uniform(30000, 35000, 100),
        'high': np.random.uniform(31000, 36000, 100),
        'low': np.random.uniform(29000, 34000, 100),
        'close': np.random.uniform(30000, 35000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
