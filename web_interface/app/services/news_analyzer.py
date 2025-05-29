from typing import Dict, List
import aiohttp
import asyncio
import json
import logging
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class NewsAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sources = [
            'binance', 'coindesk', 'cointelegraph',
            'decrypt', 'theblock', 'bitcoinmagazine',
            'cryptoslate', 'cryptobriefing', 'newsbtc',
            'ambcrypto', 'cryptopotato', 'bitcoinist'
        ]
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.cache = {}
        self.impact_threshold = 0.7

    async def analyze_news(self) -> Dict:
        try:
            # Collecte des news
            raw_news = await self._fetch_news()
            
            # Analyse du sentiment
            sentiment_results = await self._analyze_sentiment(raw_news)
            
            # Calcul de l'impact
            impact_scores = self._calculate_impact(sentiment_results)
            
            # Filtrage des news importantes
            important_news = self._filter_important_news(
                raw_news,
                sentiment_results,
                impact_scores
            )
            
            return {
                "status": "success",
                "sentiment_summary": self._generate_summary(sentiment_results),
                "important_news": important_news,
                "impact_scores": impact_scores,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            self.logger.error(f"News analysis error: {e}")
            return {"status": "error", "reason": str(e)}

    async def _fetch_news(self) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.sources:
                tasks.append(self._fetch_source_news(session, source))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            news_items = []
            for result in results:
                if isinstance(result, list):
                    news_items.extend(result)
            
            return news_items

    async def _fetch_source_news(self,
                               session: aiohttp.ClientSession,
                               source: str) -> List[Dict]:
        try:
            # URL de l'API pour chaque source
            api_url = f"https://api.{source}.com/v1/news"  # Exemple
            
            async with session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_news_data(data, source)
                else:
                    self.logger.warning(f"Failed to fetch news from {source}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error fetching from {source}: {e}")
            return []

    async def _analyze_sentiment(self, news_items: List[Dict]) -> List[Dict]:
        results = []
        
        for item in news_items:
            try:
                # Préparation du texte
                text = f"{item['title']} {item['summary']}"
                
                # Tokenization
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Prédiction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    
                # Extraction des scores
                sentiment_scores = {
                    "positive": float(probabilities[0][0]),
                    "negative": float(probabilities[0][1]),
                    "neutral": float(probabilities[0][2])
                }
                
                results.append({
                    "news_id": item["id"],
                    "sentiment": sentiment_scores,
                    "dominant_sentiment": max(sentiment_scores.items(), key=lambda x: x[1])[0]
                })
                
            except Exception as e:
                self.logger.error(f"Sentiment analysis error for news {item.get('id')}: {e}")
                continue
        
        return results

    def _calculate_impact(self, sentiment_results: List[Dict]) -> Dict:
        try:
            impact_scores = {}
            
            for result in sentiment_results:
                sentiment = result["sentiment"]
                
                # Calcul du score d'impact
                impact_score = (
                    abs(sentiment["positive"] - sentiment["negative"]) *
                    (1 - sentiment["neutral"])
                )
                
                impact_scores[result["news_id"]] = {
                    "score": impact_score,
                    "sentiment": result["dominant_sentiment"]
                }
            
            return impact_scores
            
        except Exception as e:
            self.logger.error(f"Impact calculation error: {e}")
            return {}

    def _filter_important_news(self,
                             raw_news: List[Dict],
                             sentiment_results: List[Dict],
                             impact_scores: Dict) -> List[Dict]:
        try:
            important_news = []
            
            for news in raw_news:
                news_id = news["id"]
                if news_id in impact_scores:
                    impact = impact_scores[news_id]
                    
                    if impact["score"] >= self.impact_threshold:
                        important_news.append({
                            "id": news_id,
                            "title": news["title"],
                            "source": news["source"],
                            "timestamp": news["timestamp"],
                            "impact_score": impact["score"],
                            "sentiment": impact["sentiment"],
                            "url": news["url"]
                        })
            
            return sorted(
                important_news,
                key=lambda x: x["impact_score"],
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"News filtering error: {e}")
            return []

    def _generate_summary(self, sentiment_results: List[Dict]) -> Dict:
        try:
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for result in sentiment_results:
                if result["dominant_sentiment"] == "positive":
                    positive_count += 1
                elif result["dominant_sentiment"] == "negative":
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = len(sentiment_results)
            if total == 0:
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.0
                }
            
            sentiment_scores = {
                "positive": positive_count / total,
                "negative": negative_count / total,
                "neutral": neutral_count / total
            }
            
            dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            confidence = sentiment_scores[dominant_sentiment]
            
            return {
                "overall_sentiment": dominant_sentiment,
                "confidence": confidence,
                "distribution": sentiment_scores
            }
            
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.0
            }
