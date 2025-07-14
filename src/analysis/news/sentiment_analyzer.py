# src/analysis/news/sentiment_analyzer.py
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import aiohttp
import ssl
import asyncio
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class NewsSentimentAnalyzer:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Modèle FinBERT (chargement lazy)
        self._model = None
        self._tokenizer = None

        # Configuration des sources
        self.sources = [
            {
                "name": "CoinDesk",
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "type": "rss",
                "weight": 0.9,
            },
            {
                "name": "CryptoCompare",
                "url": "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
                "type": "json",
                "weight": 0.7,
            },
            {
                "name": "Cointelegraph",
                "url": "https://cointelegraph.com/rss",
                "type": "rss",
                "weight": 0.8,
            },
        ]

        # Mapping des symboles
        self.symbol_mapping = config.get(
            "symbol_mapping",
            {
                "bitcoin": "BTC",
                "ethereum": "ETH",
                "btc": "BTC",
                "eth": "ETH",
                "cardano": "ADA",
                "solana": "SOL",
            },
        )

        # Buffer de news
        self.news_buffer: List[Dict] = []
        self.sentiment_weight = config.get("news", {}).get("sentiment_weight", 0.15)
        self.update_interval = config.get("news", {}).get("update_interval", 300)

    @property
    def model(self):
        """Chargement lazy du modèle FinBERT"""
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            )
        return self._model

    @property
    def tokenizer(self):
        """Chargement lazy du tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        return self._tokenizer

    async def fetch_all_news(self) -> List[Dict]:
        """Récupère les news de toutes les sources configurées"""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context), headers=headers
        ) as session:
            tasks = [self._fetch_source(session, source) for source in self.sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_news = []
            for result in results:
                if isinstance(result, list):
                    valid_news.extend(result)

            return valid_news

    async def _fetch_source(
        self, session: aiohttp.ClientSession, source: Dict
    ) -> List[Dict]:
        """Récupère et parse les news d'une source spécifique"""
        try:
            async with session.get(source["url"], timeout=30) as response:
                if response.status != 200:
                    return []

                if source["type"] == "rss":
                    content = await response.text()
                    return self._parse_rss(content, source)
                else:
                    data = await response.json()
                    return self._parse_json(data, source)
        except Exception as e:
            self.logger.error(f"Error fetching {source['name']}: {str(e)}")
            return []

    def _parse_rss(self, content: str, source: Dict) -> List[Dict]:
        """Parse le contenu RSS"""
        try:
            soup = BeautifulSoup(content, "xml")
            items = soup.find_all("item")
            return [self._parse_rss_item(item, source) for item in items]
        except Exception as e:
            self.logger.error(f"Error parsing RSS {source['name']}: {str(e)}")
            return []

    def _parse_rss_item(self, item, source: Dict) -> Dict:
        """Parse un item RSS individuel"""
        return {
            "title": item.find("title").text if item.find("title") else "",
            "text": item.find("description").text if item.find("description") else "",
            "source": source["name"],
            "timestamp": self._parse_timestamp(item),
            "url": item.find("link").text if item.find("link") else "",
            "symbols": self._extract_symbols(item),
            "source_weight": source["weight"],
        }

    def analyze_sentiment_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Analyse le sentiment par batch pour meilleure performance"""
        if not news_items:
            return []

        try:
            # Préparation des textes
            texts = [f"{item['title']}. {item['text']}"[:512] for item in news_items]

            # Tokenization par batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Formatage des résultats
            results = []
            for i, item in enumerate(news_items):
                sentiment = float(scores[i][1] - scores[i][0])  # Pos - Neg
                results.append(
                    {
                        **item,
                        "sentiment": sentiment,
                        "impact_score": self._calculate_impact(item, sentiment),
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return []

    async def update_analysis(self):
        """Mise à jour complète de l'analyse"""
        try:
            # 1. Récupération des news
            raw_news = await self.fetch_all_news()

            # 2. Analyse de sentiment
            analyzed_news = self.analyze_sentiment_batch(raw_news)

            # 3. Mise à jour du buffer (garder les 200 plus récentes)
            self.news_buffer = [
                *self.news_buffer[-100:],  # Garde les 100 précédentes
                *analyzed_news,
            ][
                -200:
            ]  # Limite totale à 200

            # 4. Sauvegarde de l'état
            await self._save_state()

            return analyzed_news

        except Exception as e:
            self.logger.error(f"Error in news update: {str(e)}")
            return []

    async def get_symbol_sentiment(self, symbol: str) -> float:
        """Récupère le sentiment pondéré pour un symbole spécifique"""
        try:
            symbol_key = symbol.replace("/", "").upper()
            total = 0.0
            total_weight = 0.0
            current_time = datetime.now().timestamp()

            for news in self.news_buffer:
                if symbol_key in news.get("symbols", []):
                    # Décay exponentiel basé sur l'âge (50% decay après 24h)
                    hours_old = (
                        current_time - news.get("timestamp", current_time)
                    ) / 3600
                    decay = 0.5 ** (hours_old / 24)

                    total += news["sentiment"] * news["impact_score"] * decay
                    total_weight += news["impact_score"] * decay

            return total / max(total_weight, 1e-6)  # Évite division par zéro

        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return 0.0

    async def _save_state(self, path: Optional[str] = None):
        """Sauvegarde l'état courant du analyseur"""
        path = path or self.config.get("news", {}).get(
            "storage_path", "data/news_analysis.json"
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {
                        "last_updated": datetime.now().isoformat(),
                        "news_count": len(self.news_buffer),
                        "symbol_mapping": self.symbol_mapping,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")

    # ... (autres méthodes utilitaires comme _parse_timestamp, _extract_symbols, etc.)
