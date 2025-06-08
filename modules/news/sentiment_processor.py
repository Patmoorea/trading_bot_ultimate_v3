try:
    from transformers import pipeline
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import transformers - {str(e)}")
    from tests.mocks.transformers import pipeline

class NewsProcessor:
    def __init__(self):
        self.analyzer = pipeline('sentiment-analysis')
    
    def analyze(self, text):
        return self.analyzer(text)
