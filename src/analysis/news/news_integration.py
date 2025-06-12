import aiohttp
import pandas as pd
from transformers import pipeline
from typing import List, Dict, Any
import logging

class NewsAnalyzer:
    def __init__(self):
        self.sources = [
            'cryptopanic.com',
            'coindesk.com',
            'cointelegraph.com',
            'bitcoinmagazine.com',
            'decrypt.co',
            'theblockcrypto.com',
            'newsbtc.com',
            'bitcoinist.com',
            'cryptoslate.com',
            'cryptobriefing.com',
            'ambcrypto.com',
            'beincrypto.com'
        ]
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        self.logger = logging.getLogger(__name__)

    async def fetch_news(self) -> List[Dict[str, Any]]:
        """Récupère les news de toutes les sources"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_source(session, source) for source in self.sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        news_items = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Erreur récupération news: {str(result)}")
            else:
                news_items.extend(result)
                
        return news_items
    
    def analyze_sentiment(self, news_items: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyse le sentiment des news avec FinBERT"""
        texts = [item['title'] + ' ' + item['description'] for item in news_items]
        sentiments = self.sentiment_model(texts)
        
        df = pd.DataFrame(news_items)
        df['sentiment_score'] = [s['score'] for s in sentiments]
        df['sentiment'] = [s['label'] for s in sentiments]
        
        return df
    
    def calculate_market_impact(self, analyzed_news: pd.DataFrame) -> Dict[str, float]:
        """Calcule l'impact potentiel sur le marché"""
        impact = {
            'bullish_score': 0.0,
            'bearish_score': 0.0,
            'neutral_score': 0.0
        }
        
        # Pondération basée sur la source et le sentiment
        for _, news in analyzed_news.iterrows():
            weight = self._calculate_source_weight(news['source'])
            if news['sentiment'] == 'positive':
                impact['bullish_score'] += news['sentiment_score'] * weight
            elif news['sentiment'] == 'negative':
                impact['bearish_score'] += news['sentiment_score'] * weight
            else:
                impact['neutral_score'] += news['sentiment_score'] * weight
                
        return impact
    
    def _calculate_source_weight(self, source: str) -> float:
        """Calcule le poids de crédibilité d'une source"""
        weights = {
            'coindesk.com': 1.0,
            'cointelegraph.com': 0.9,
            'theblockcrypto.com': 0.95,
            'bitcoinmagazine.com': 0.85
        }
        return weights.get(source, 0.7)  # 0.7 par défaut
