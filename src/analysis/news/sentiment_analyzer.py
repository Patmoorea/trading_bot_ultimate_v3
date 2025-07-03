from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict
import aiohttp
import asyncio
import json
from datetime import datetime


class NewsSentimentAnalyzer:
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sources = self._init_sources()

    def _init_sources(self) -> List[Dict]:
        """Initialise les sources d'actualités"""
        return [
            {"name": "CoinDesk", "url": "https://api.coindesk.com/v1/news"},
            {
                "name": "CryptoCompare",
                "url": "https://min-api.cryptocompare.com/data/v2/news",
            },
            {"name": "Cointelegraph", "url": "https://cointelegraph.com/api/v1/news"},
        ]
    def analyze_news(data):
        if isinstance(data, list):
            for d in data:
                if isinstance(d, dict):
                    titre = d.get("title", "")
                    # suite du traitement...
                else:
                    print("Élément news non dict :", type(d))
        elif isinstance(data, dict):
            # traitement direct
        else:
            print("Format inattendu :", type(data))
        
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

    def _parse_news(self, data: Dict, source: str) -> List[Dict]:
        """Parse les news selon la source et retourne un format standardisé"""
        parsed_news = []
        try:
            if source == "CoinDesk":
                news_items = data.get("news", [])
                for item in news_items:
                    parsed_news.append(
                        {
                            "title": item.get("title", ""),
                            "text": item.get("description", ""),
                            "source": source,
                            "timestamp": item.get("published_at", ""),
                            "url": item.get("url", ""),
                            "symbols": self._extract_crypto_symbols(
                                item.get("tags", [])
                            ),
                        }
                    )

            elif source == "CryptoCompare":
                news_items = data.get("Data", [])
                for item in news_items:
                    parsed_news.append(
                        {
                            "title": item.get("title", ""),
                            "text": item.get("body", ""),
                            "source": source,
                            "timestamp": item.get("published_on", ""),
                            "url": item.get("url", ""),
                            "symbols": self._extract_crypto_symbols(
                                item.get("categories", "").split("|")
                            ),
                        }
                    )

            elif source == "Cointelegraph":
                news_items = data.get("data", [])
                for item in news_items:
                    parsed_news.append(
                        {
                            "title": item.get("title", ""),
                            "text": f"{item.get('title', '')}. {item.get('description', '')}",
                            "source": source,
                            "timestamp": item.get("publishedAt", ""),
                            "url": item.get("url", ""),
                            "symbols": self._extract_crypto_symbols(
                                item.get("tags", [])
                            ),
                        }
                    )

            # Ajouter l'horodatage UTC si manquant
            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            for item in parsed_news:
                if not item["timestamp"]:
                    item["timestamp"] = current_time

            return parsed_news

        except Exception as e:
            print(f"Erreur parsing news {source}: {e}")
            return []

    def _extract_crypto_symbols(self, tags: List[str]) -> List[str]:
        """Extrait les symboles de crypto des tags"""
        crypto_mapping = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "eth"],
            "USDC": ["usdc", "usd coin"],
            "BNB": ["binance", "bnb"],
            "XRP": ["ripple", "xrp"],
            "ADA": ["cardano", "ada"],
            "DOGE": ["dogecoin", "doge"],
        }

        found_symbols = set()
        for tag in tags:
            tag = tag.lower()
            for symbol, keywords in crypto_mapping.items():
                if any(keyword in tag for keyword in keywords):
                    found_symbols.add(symbol)

        return list(found_symbols)

    def analyze_sentiment(self, news: List[Dict]) -> List[Dict]:
        """Analyse le sentiment avec FinBERT"""
        results = []
        for item in news:
            try:
                # Tokenization
                inputs = self.tokenizer(
                    item["text"], return_tensors="pt", truncation=True, max_length=512
                )
                # Prédiction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                # Score final
                sentiment_score = float(
                    scores[0][1] - scores[0][0]
                )  # positive - negative
                results.append(
                    {
                        "title": item["title"],
                        "source": item["source"],
                        "sentiment": sentiment_score,
                        "impact_score": self._calculate_impact(sentiment_score, item),
                        "timestamp": item["timestamp"],
                        "symbols": item.get("symbols", []),
                    }
                )
            except Exception as e:
                print(f"Erreur analyse sentiment: {e}")
                continue
        return results

    def _calculate_impact(self, sentiment_score: float, news: Dict) -> float:
        """Calcule le score d'impact (0-1)"""
        try:
            # Facteurs de pondération
            source_weight = self._get_source_weight(news["source"])
            time_weight = self._get_time_weight(news["timestamp"])
            relevance_weight = self._get_relevance_weight(news["title"])

            # Score final
            impact = (
                abs(sentiment_score) * 0.4
                + source_weight * 0.3
                + time_weight * 0.2
                + relevance_weight * 0.1
            )
            return min(max(impact, 0), 1)  # Normalisation 0-1
        except Exception as e:
            print(f"Erreur calcul impact: {e}")
            return 0.5

    def _get_source_weight(self, source: str) -> float:
        """Retourne le poids de crédibilité de la source"""
        weights = {"CoinDesk": 0.9, "CryptoCompare": 0.7, "Cointelegraph": 0.8}
        return weights.get(source, 0.5)

    def _get_time_weight(self, timestamp: str) -> float:
        """Calcule le poids basé sur l'âge de la news"""
        try:
            # Convertir le timestamp en datetime
            news_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            age_hours = (datetime.utcnow() - news_time).total_seconds() / 3600

            # Décroissance exponentielle sur 24h
            return max(0.1, np.exp(-age_hours / 24))
        except:
            return 0.5

    def _get_relevance_weight(self, title: str) -> float:
        """Calcule le poids basé sur la pertinence du titre"""
        keywords = [
            "bitcoin",
            "ethereum",
            "crypto",
            "blockchain",
            "market",
            "trade",
            "price",
            "analysis",
            "btc",
            "eth",
            "defi",
            "nft",
        ]

        title_lower = title.lower()
        keyword_count = sum(1 for keyword in keywords if keyword in title_lower)

        # Normalisation entre 0.1 et 1.0
        return min(1.0, 0.1 + (keyword_count * 0.2))
