import os
from typing import Dict, List
import aiohttp
import logging
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_LANGUAGES = os.getenv("NEWS_API_LANGUAGES", "en").split(",")
NEWS_SOURCES = os.getenv("NEWS_SOURCES", "").split(",")
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.7"))


class NewsAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )
        self.impact_threshold = SENTIMENT_THRESHOLD

    async def analyze_news(self) -> Dict:
        try:
            # 1. Récupération des news (CryptoPanic prioritaire, fallback NewsAPI)
            news = await self._fetch_cryptopanic_news()
            if not news and NEWS_API_KEY:
                self.logger.info("CryptoPanic vide, fallback sur NewsAPI")
                news = await self._fetch_newsapi_news()
            if not news:
                return {
                    "status": "error",
                    "reason": "Aucune news récupérée sur aucune source",
                }

            # 2. Analyse du sentiment
            sentiment_results = await self._analyze_sentiment(news)

            # 3. Calcul de l'impact
            impact_scores = self._calculate_impact(sentiment_results)

            # 4. Filtrage des news importantes
            important_news = self._filter_important_news(
                news, sentiment_results, impact_scores
            )

            return {
                "status": "success",
                "sentiment_summary": self._generate_summary(sentiment_results),
                "important_news": important_news,
                "impact_scores": impact_scores,
                "timestamp": datetime.now(timezone.utc),
            }
        except Exception as e:
            self.logger.error(f"News analysis error: {e}")
            return {"status": "error", "reason": str(e)}

    async def _fetch_cryptopanic_news(self) -> List[Dict]:
        if not CRYPTO_PANIC_API_KEY:
            return []
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&public=true"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        news_items = []
                        for item in data.get("results", []):
                            news_items.append(
                                {
                                    "id": item.get("id", ""),
                                    "title": item.get("title", ""),
                                    "summary": item.get("body", ""),
                                    "url": item.get("url", ""),
                                    "published_at": item.get("published_at", ""),
                                    "source": item.get("source", {}).get(
                                        "title", "CryptoPanic"
                                    ),
                                }
                            )
                        return news_items
                    else:
                        self.logger.error(
                            f"Erreur CryptoPanic: Status code {resp.status}"
                        )
        except Exception as e:
            self.logger.error(f"Erreur de récupération CryptoPanic: {e}")
        return []

    async def _fetch_newsapi_news(self) -> List[Dict]:
        if not NEWS_API_KEY:
            return []
        try:
            base_url = "https://newsapi.org/v2/top-headlines"
            params = {
                "apiKey": NEWS_API_KEY,
                "language": NEWS_API_LANGUAGES[
                    0
                ],  # NewsAPI ne prend qu'une langue à la fois
                "pageSize": 50,
            }
            if NEWS_SOURCES and NEWS_SOURCES[0]:
                params["sources"] = ",".join(NEWS_SOURCES)
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        news_items = []
                        for i, item in enumerate(data.get("articles", [])):
                            news_items.append(
                                {
                                    "id": f"newsapi_{i}",
                                    "title": item.get("title", ""),
                                    "summary": item.get("description", ""),
                                    "url": item.get("url", ""),
                                    "published_at": item.get("publishedAt", ""),
                                    "source": item.get("source", {}).get(
                                        "name", "NewsAPI"
                                    ),
                                }
                            )
                        return news_items
                    else:
                        self.logger.error(f"Erreur NewsAPI: Status code {resp.status}")
        except Exception as e:
            self.logger.error(f"Erreur de récupération NewsAPI: {e}")
        return []

    async def _analyze_sentiment(self, news_items: List[Dict]) -> List[Dict]:
        results = []
        for item in news_items:
            try:
                text = f"{item['title']} {item['summary']}"
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                sentiment_labels = ["negative", "neutral", "positive"]
                sentiment_idx = torch.argmax(probabilities).item()
                confidence = probabilities[0, sentiment_idx].item()
                results.append(
                    {
                        "id": item.get("id"),
                        "sentiment": sentiment_labels[sentiment_idx],
                        "confidence": confidence,
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "url": item.get("url"),
                        "source": item.get("source"),
                        "published_at": item.get("published_at"),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Erreur analyse sentiment news: {e}")
        return results

    def _calculate_impact(self, sentiment_results: List[Dict]) -> Dict:
        # Impact simple = confiance du modèle (peut être raffiné)
        impact_scores = {}
        for r in sentiment_results:
            impact_scores[r["id"]] = r["confidence"]
        return impact_scores

    def _filter_important_news(
        self, news, sentiment_results, impact_scores
    ) -> List[Dict]:
        important = []
        for r in sentiment_results:
            if r["confidence"] >= self.impact_threshold or r["sentiment"] == "negative":
                important.append(r)
        return important

    def _generate_summary(self, sentiment_results: List[Dict]) -> Dict:
        if not sentiment_results:
            return {"overall_sentiment": "neutral", "confidence": 0.0}
        sentiments = [r["sentiment"] for r in sentiment_results]
        confidences = [r["confidence"] for r in sentiment_results]
        sentiment_counts = {
            k: sentiments.count(k) for k in ["positive", "neutral", "negative"]
        }
        overall = max(sentiment_counts, key=sentiment_counts.get)
        avg_conf = float(np.mean(confidences))
        return {"overall_sentiment": overall, "confidence": avg_conf}
