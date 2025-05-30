# src/news_processor/core.py VERSION 2.5.1 
# Last Updated: 2025-05-30 04:38:30 UTC
# Author: Patmoorea

from transformers import BertForSequenceClassification, BertTokenizer
import torch
from typing import List, Dict
import os
import json
from datetime import datetime, timedelta

class NewsProcessorBase:
    """Classe de base pour le traitement des news"""
    def __init__(self):
        self.history_path = "data/news_history.json"
        self._load_history()

    def _load_history(self):
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except Exception as e:
            print(f"Erreur chargement historique: {e}")
            self.history = []

    def _save_history(self):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            print(f"Erreur sauvegarde historique: {e}")

class NewsSentimentAnalyzer(NewsProcessorBase):
    def __init__(self):
        super().__init__()
        # Optimisation Metal pour M1/M4
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model = self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.sources = ["coindesk", "reuters", "bloomberg", "cryptopanic"]

    def analyze(self, texts: List[str]) -> List[Dict]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            results = [{
                "text": texts[i],
                "sentiment": "bullish" if torch.argmax(probs[i]).item() == 1 else "bearish",
                "confidence": torch.max(probs[i]).item(),
                "timestamp": datetime.utcnow().isoformat()
            } for i in range(len(texts))]
            
            # Sauvegarde dans l'historique
            self.history.extend(results)
            self._save_history()
            
            return results
            
        except Exception as e:
            print(f"Erreur analyse: {e}")
            return []

class CachedNewsSentimentAnalyzer(NewsSentimentAnalyzer):
    """Version optimisée avec cache local du modèle FinBERT"""
    def __init__(self, model_path: str = "./model_cache/finbert"):
        super().__init__()
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Cache non trouvé ({str(e)}), téléchargement...")
            self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model = self.model.to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

    def get_historical_sentiment(self, days: int = 7) -> Dict:
        """Analyse l'historique des sentiments"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        recent_history = [h for h in self.history if h["timestamp"] > cutoff]
        
        if not recent_history:
            return {"bullish": 0, "bearish": 0, "average_confidence": 0}
            
        bullish_count = sum(1 for h in recent_history if h["sentiment"] == "bullish")
        total = len(recent_history)
        avg_conf = sum(h["confidence"] for h in recent_history) / total
        
        return {
            "bullish": bullish_count / total,
            "bearish": (total - bullish_count) / total,
            "average_confidence": avg_conf
        }
