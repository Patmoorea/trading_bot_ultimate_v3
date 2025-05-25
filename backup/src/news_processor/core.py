from transformers import BertForSequenceClassification, BertTokenizer
import torch
from typing import List, Dict

class NewsSentimentAnalyzer:
    def __init__(self):
        # Désactive l'optimisation Metal temporairement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model = self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.sources = ["coindesk", "reuters", "bloomberg", "cryptopanic"]

    def analyze(self, texts: List[str]) -> List[Dict]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return [{
            "text": texts[i],
            "sentiment": "bullish" if torch.argmax(probs[i]).item() == 1 else "bearish",
            "confidence": torch.max(probs[i]).item()
        } for i in range(len(texts))]

class CachedNewsSentimentAnalyzer(NewsSentimentAnalyzer):
    """Version optimisée avec cache local du modèle FinBERT"""
    def __init__(self, model_path: str = "./model_cache/finbert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        except:
            print("Cache non trouvé, téléchargement du modèle...")
            self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
        self.sources = ["coindesk", "reuters", "bloomberg", "cryptopanic"]

class CachedNewsSentimentAnalyzer:
    """Version optimisée avec cache local - NE REMPLACE PAS NewsSentimentAnalyzer"""
    def __init__(self, model_path="model_cache/finbert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Chargement du cache échoué ({str(e)}), téléchargement...")
            self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
        self.sources = ["coindesk", "reuters", "bloomberg", "cryptopanic"]

    def analyze(self, texts: List[str]) -> List[Dict]:
        """Même implémentation que NewsSentimentAnalyzer mais avec cache"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return [{
            "text": texts[i],
            "sentiment": "bullish" if torch.argmax(probs[i]).item() == 1 else "bearish",
            "confidence": torch.max(probs[i]).item()
        } for i in range(len(texts))]
