"""
Mock dependencies for testing
Version 1.0.0 - Created: 2025-05-19 01:53:52 by Patmoorea
"""

class MockSentiment:
    def __init__(self):
        self.polarity = 0.5
        self.subjectivity = 0.5

class MockTextBlob:
    def __init__(self, text):
        self.sentiment = MockSentiment()

class MockPipeline:
    def __init__(self, task=None, **kwargs):
        self.task = task

    def __call__(self, text):
        return [{'label': 'positive', 'score': 0.85}]

def mock_pipeline(*args, **kwargs):
    return MockPipeline()
