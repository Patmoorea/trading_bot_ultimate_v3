import feedparser
from transformers import pipeline
import warnings

# Suppression des avertissements de watermark
warnings.filterwarnings("ignore", message=".*watermark ratio.*")

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.analyzer = pipeline(
                "text-classification",
                model="finiteautomata/bertweet-base-sentiment-analysis",
                device="mps"  # Pour Apple Silicon
            )
        except Exception as e:
            print(f"Failed to initialize analyzer: {e}")
            self.analyzer = None
        
    def analyze(self, text):
        if not self.analyzer:
            return {'sentiment': 'neutral', 'score': 0.5}
            
        try:
            result = self.analyzer(text[:512])  # Truncate to model max length
            return {
                'sentiment': result[0]['label'],
                'score': result[0]['score']
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'sentiment': 'neutral', 'score': 0.5}
