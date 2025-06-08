from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict
import aiohttp
import asyncio
import json

class NewsSentimentAnalyzer:
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sources = self._init_sources()
        
    def _init_sources(self) -> List[Dict]:
        """Initialise les 12 sources d'actualités"""
        return [
            {"name": "CoinDesk", "url": "https://api.coindesk.com/v1/news"},
            {"name": "CryptoCompare", "url": "https://min-api.cryptocompare.com/data/v2/news"},
            {"name": "Cointelegraph", "url": "https://cointelegraph.com/api/v1/news"},
            # ... autres sources
        ]
        
    async def fetch_all_news(self) -> List[Dict]:
        """Récupère les news de toutes les sources"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.sources:
                tasks.append(self.fetch_news(session, source))
            return await asyncio.gather(*tasks)
            
    async def fetch_news(self, session, source: Dict) -> List[Dict]:
        """Récupère les news d'une source"""
        try:
            async with session.get(source["url"]) as response:
                data = await response.json()
                return self._parse_news(data, source["name"])
        except Exception as e:
            print(f"Erreur récupération news {source['name']}: {str(e)}")
            return []
            
    def analyze_sentiment(self, news: List[Dict]) -> List[Dict]:
        """Analyse le sentiment avec FinBERT"""
        results = []
        for item in news:
            # Tokenization
            inputs = self.tokenizer(
                item["text"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                
            # Score final
            sentiment_score = float(scores[0][1] - scores[0][0])  # positive - negative
            
            results.append({
                "title": item["title"],
                "source": item["source"],
                "sentiment": sentiment_score,
                "impact_score": self._calculate_impact(sentiment_score, item),
                "timestamp": item["timestamp"]
            })
            
        return results
        
    def _calculate_impact(self, sentiment_score: float, news: Dict) -> float:
        """Calcule le score d'impact (0-1)"""
        # Facteurs de pondération
        source_weight = self._get_source_weight(news["source"])
        time_weight = self._get_time_weight(news["timestamp"])
        relevance_weight = self._get_relevance_weight(news["title"])
        
        # Score final
        impact = (
            abs(sentiment_score) * 0.4 +
            source_weight * 0.3 +
            time_weight * 0.2 +
            relevance_weight * 0.1
        )
        
        return min(max(impact, 0), 1)  # Normalisation 0-1
