from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict
import aiohttp
import asyncio
import json
from datetime import datetime
import logging
import requests
from bs4 import BeautifulSoup
import ssl
from time import mktime


class NewsSentimentAnalyzer:
    def __init__(self):
        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Configuration du modèle FinBERT
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Headers HTTP
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # Sources de news
        self.sources = self._init_sources()

    def _init_sources(self) -> List[Dict]:
        """Initialise les sources d'actualités avec les flux RSS"""
        return [
            {
                "name": "CoinDesk",
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "type": "rss",
            },
            {
                "name": "CryptoCompare",
                "url": "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
                "type": "json",
            },
            {
                "name": "Cointelegraph",
                "url": "https://cointelegraph.com/rss",
                "type": "rss",
            },
        ]

    async def fetch_all_news(self) -> List[Dict]:
        """Récupère les news de toutes les sources"""
        # Configuration SSL pour éviter les erreurs de certificat
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(
            connector=connector, headers=self.headers
        ) as session:
            tasks = []
            for source in self.sources:
                tasks.append(self.fetch_news(session, source))

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                valid_results = []
                for result in results:
                    if isinstance(result, list):
                        valid_results.extend(result)
                    elif isinstance(result, Exception):
                        self.logger.error(f"Erreur fetch: {str(result)}")
                return valid_results
            except Exception as e:
                self.logger.error(f"Erreur fetch_all_news: {str(e)}")
                return []

    def _parse_rss_feed(self, content, source: str) -> List[Dict]:
        """Parse un flux RSS"""
        parsed_news = []
        try:
            # Vérification que content n'est pas None
            if not content:
                self.logger.error(f"Contenu vide pour {source}")
                return []

            # Parsing avec gestion d'erreur explicite
            try:
                soup = BeautifulSoup(content, "xml")
            except Exception as parse_error:
                self.logger.error(f"Erreur parsing XML pour {source}: {parse_error}")
                return []

            # Vérification de la structure RSS
            if not soup.find("rss"):
                self.logger.error(f"Format RSS invalide pour {source}")
                return []

            items = soup.find_all("item")
            if not items:
                self.logger.error(f"Aucun item trouvé pour {source}")
                return []

            for item in items:
                try:
                    # Extraction sécurisée des données
                    title = item.find("title")
                    description = item.find("description")
                    link = item.find("link")
                    pub_date = item.find("pubDate")

                    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    if pub_date and pub_date.text:
                        try:
                            from email.utils import parsedate_to_datetime

                            timestamp = parsedate_to_datetime(pub_date.text).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        except Exception as date_error:
                            self.logger.warning(
                                f"Erreur parsing date pour {source}: {date_error}"
                            )

                    news_item = {
                        "title": title.text if title else "",
                        "text": description.text if description else "",
                        "source": source,
                        "timestamp": timestamp,
                        "url": link.text if link else "",
                        "symbols": self._extract_crypto_symbols(
                            [tag.text for tag in item.find_all("category")]
                            if item.find_all("category")
                            else []
                        ),
                    }
                    parsed_news.append(news_item)

                except Exception as item_error:
                    self.logger.warning(
                        f"Erreur parsing item pour {source}: {item_error}"
                    )
                    continue

            return parsed_news

        except Exception as e:
            self.logger.error(f"Erreur parsing RSS {source}: {str(e)}")
            return []

    async def fetch_news(
        self, session: aiohttp.ClientSession, source: Dict
    ) -> List[Dict]:
        """Récupère les news de toutes les sources"""
        all_news = []

        # Configuration SSL pour éviter les erreurs de certificat
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        try:
            async with aiohttp.ClientSession(
                connector=connector, headers=self.headers
            ) as session:
                for source in self.sources:
                    try:
                        async with session.get(source["url"], timeout=30) as response:
                            if response.status != 200:
                                self.logger.error(
                                    f"Erreur HTTP {response.status} pour {source['name']}"
                                )
                                continue

                            content = await response.text()

                            if source["type"] == "rss":
                                news = self._parse_rss_feed(content, source["name"])
                            else:
                                data = await response.json()
                                news = self._parse_news(data, source["name"])

                            if news:
                                all_news.extend(news)

                    except Exception as e:
                        self.logger.error(
                            f"Erreur récupération news pour {source['name']}: {e}"
                        )
                        continue

            return all_news

        except Exception as e:
            self.logger.error(f"Erreur fetch_news: {e}")
            return []

    def _parse_news(self, data: Dict, source: str) -> List[Dict]:
        """Parse les données JSON des APIs"""
        parsed_news = []
        try:
            if source == "CryptoCompare":
                news_items = data.get("Data", [])
                for item in news_items:
                    parsed_news.append(
                        {
                            "title": item.get("title", ""),
                            "text": item.get("body", ""),
                            "source": source,
                            "timestamp": datetime.fromtimestamp(
                                item.get("published_on", datetime.utcnow().timestamp())
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                            "url": item.get("url", ""),
                            "symbols": self._extract_crypto_symbols(
                                item.get("categories", "").split("|")
                            ),
                        }
                    )

            return parsed_news
        except Exception as e:
            self.logger.error(f"Erreur parsing news {source}: {str(e)}")
            return []

    def _extract_crypto_symbols(self, tags: List[str]) -> List[str]:
        """Extrait les symboles de crypto des tags"""
        crypto_mapping = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "eth"],
            "USDT": ["usdt", "usdt"],
            "BNB": ["binance", "bnb"],
            "XRP": ["ripple", "xrp"],
            "ADA": ["cardano", "ada"],
            "DOGE": ["dogecoin", "doge"],
        }

        found_symbols = set()
        for tag in tags:
            tag = str(tag).lower()
            for symbol, keywords in crypto_mapping.items():
                if any(keyword in tag for keyword in keywords):
                    found_symbols.add(symbol)

        return list(found_symbols)

    def analyze_sentiment(self, news: List[Dict]) -> List[Dict]:
        """Analyse le sentiment avec FinBERT"""
        results = []
        for item in news:
            try:
                inputs = self.tokenizer(
                    item["text"], return_tensors="pt", truncation=True, max_length=512
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    scores = torch.nn.functional.softmax(outputs.logits, dim=1)

                sentiment_score = float(scores[0][1] - scores[0][0])

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
                self.logger.error(f"Erreur analyse sentiment: {str(e)}")
                continue

        return results

    def _calculate_impact(self, sentiment_score: float, news: Dict) -> float:
        """Calcule le score d'impact (0-1)"""
        try:
            source_weight = self._get_source_weight(news["source"])
            time_weight = self._get_time_weight(news["timestamp"])
            relevance_weight = self._get_relevance_weight(news["title"])

            impact = (
                abs(sentiment_score) * 0.4
                + source_weight * 0.3
                + time_weight * 0.2
                + relevance_weight * 0.1
            )
            return min(max(impact, 0), 1)
        except Exception as e:
            self.logger.error(f"Erreur calcul impact: {str(e)}")
            return 0.5

    def _get_source_weight(self, source: str) -> float:
        """Retourne le poids de crédibilité de la source"""
        weights = {"CoinDesk": 0.9, "CryptoCompare": 0.7, "Cointelegraph": 0.8}
        return weights.get(source, 0.5)

    def _get_time_weight(self, timestamp: str) -> float:
        """Calcule le poids basé sur l'âge de la news"""
        try:
            news_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            age_hours = (datetime.utcnow() - news_time).total_seconds() / 3600
            return max(0.1, np.exp(-age_hours / 24))
        except Exception:
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
        return min(1.0, 0.1 + (keyword_count * 0.2))
