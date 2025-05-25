class NewsSentimentAnalyzer:
    def __init__(self, model_path=None):
        """Analyseur de sentiment avec pipeline configurable"""
        self.sentiment_pipeline = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Charge le modèle de sentiment"""
        # Implémentation simplifiée pour les tests
        return lambda text: {
            'positive': 0.8 if "good" in text.lower() else 0.2,
            'negative': 0.2 if "good" in text.lower() else 0.8
        }
