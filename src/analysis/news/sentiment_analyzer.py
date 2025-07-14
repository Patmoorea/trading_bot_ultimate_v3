import os
import json
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Optional, Set
import logging
import aiohttp
import ssl
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class SymbolExtractor:
    """
    Extraction avancée des symboles crypto depuis les titres et textes des news.
    Utilise un mapping regex et analyse contextuelle (BTC, ETH, SOL, etc).
    """

    def __init__(self, symbol_mapping: Optional[Dict[str, str]] = None):
        self.symbol_mapping = symbol_mapping or {
            "bitcoin": "BTC",
            "ethereum": "ETH",
            "btc": "BTC",
            "eth": "ETH",
            "cardano": "ADA",
            "solana": "SOL",
            "sol": "SOL",
            "ada": "ADA",
            "ripple": "XRP",
            "xrp": "XRP",
            "dogecoin": "DOGE",
            "doge": "DOGE",
            "polkadot": "DOT",
            "dot": "DOT",
            "binance": "BNB",
            "bnb": "BNB",
            "matic": "MATIC",
            "polygon": "MATIC",
            "litecoin": "LTC",
            "ltc": "LTC",
            "shiba": "SHIB",
            "shib": "SHIB",
            "tron": "TRX",
            "trx": "TRX",
            "avalanche": "AVAX",
            "avax": "AVAX",
            "chainlink": "LINK",
            "link": "LINK",
            "uniswap": "UNI",
            "uni": "UNI",
            "stellar": "XLM",
            "xlm": "XLM",
            "vechain": "VET",
            "vet": "VET",
            "aptos": "APT",
            "apt": "APT",
            "arbitrum": "ARB",
            "arb": "ARB",
            "optimism": "OP",
            "op": "OP",
            "the sandbox": "SAND",
            "sand": "SAND",
            "decentraland": "MANA",
            "mana": "MANA",
            # Ajoute d'autres ici selon besoin
        }
        # Compile regex patterns for all known tickers/names (mot entier uniquement)
        self.regex_patterns = [
            (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), ticker)
            for name, ticker in self.symbol_mapping.items()
        ]

    def extract_symbols(self, text: str) -> List[str]:
        """
        Extraction robuste des symboles à partir d'un texte (titre+contenu).
        - Mapping regex sur tous les mots clés et tickers connus.
        - Extraction contextuelle des paires du style BTC/USDT, ETHUSDT, $BTC, etc.
        """
        if not text:
            return []
        found: Set[str] = set()

        # 1. Mapping regex sur tous les noms connus
        for pattern, ticker in self.regex_patterns:
            if pattern.search(text):
                found.add(ticker)

        # 2. Extraction contextuelle : paires du type BTC/USDT, ETHUSDT, $DOGE, etc.
        for match in re.findall(
            r"\b([A-Z]{2,6})(?:[-/])?(USDT|USD|EUR|BTC)?\b", text.upper()
        ):
            ticker = match[0]
            if ticker not in {"USD", "USDT", "EUR", "BTC"} and len(ticker) >= 3:
                found.add(ticker)
        for match in re.findall(r"\$([A-Z]{2,6})\b", text.upper()):
            if match not in {"USD", "USDT", "EUR", "BTC"} and len(match) >= 3:
                found.add(match)
        return list(found)


class NewsSentimentAnalyzer:
    """
    Analyseur avancé des news crypto avec extraction robuste des symboles et analyse de sentiment.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        self.symbol_extractor = SymbolExtractor(config.get("symbol_mapping"))

        # Modèle FinBERT (lazy)
        self._model = None
        self._tokenizer = None

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

        self.news_buffer: List[Dict] = []
        self.sentiment_weight = config.get("news", {}).get("sentiment_weight", 0.15)
        self.update_interval = config.get("news", {}).get("update_interval", 300)

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        return self._tokenizer

    async def fetch_all_news(self) -> List[Dict]:
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

            print(f"\n[NEWS DEBUG] {len(valid_news)} news récupérées ce cycle")
            if valid_news:
                for idx, n in enumerate(valid_news[:5]):
                    print(
                        f"  - {n['source']}: {n['title'][:100]} | Symbols: {n['symbols']}"
                    )
            else:
                print("  (Aucune news récupérée)")
            return valid_news

    async def _fetch_source(
        self, session: aiohttp.ClientSession, source: Dict
    ) -> List[Dict]:
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
        try:
            soup = BeautifulSoup(content, "xml")
            items = soup.find_all("item")
            return [self._parse_rss_item(item, source) for item in items]
        except Exception as e:
            self.logger.error(f"Error parsing RSS {source['name']}: {str(e)}")
            return []

    def _parse_rss_item(self, item, source: Dict) -> Dict:
        title = item.find("title").text if item.find("title") else ""
        description = item.find("description").text if item.find("description") else ""
        url = item.find("link").text if item.find("link") else ""
        # Extraction avancée des symboles (titre + description)
        symbols = self.symbol_extractor.extract_symbols(f"{title} {description}")
        return {
            "title": title,
            "text": description,
            "source": source["name"],
            "timestamp": self._parse_timestamp(item),
            "url": url,
            "symbols": symbols,
            "source_weight": source["weight"],
        }

    def analyze_sentiment_batch(self, news_items: List[Dict]) -> List[Dict]:
        print("[DEBUG] analyze_sentiment_batch entrée")
        if not news_items:
            print("[SENTIMENT] Aucun article à analyser.")
            return []

        try:
            print("[DEBUG] Après try, news_items size:", len(news_items))
            texts = [f"{item['title']}. {item['text']}"[:512] for item in news_items]
            print("[DEBUG] Nombre de textes à analyser:", len(texts))
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            print("[DEBUG] Inputs batch prêt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            print("[DEBUG] Softmax scores:", scores.tolist()[:5])

            results = []
            for i, item in enumerate(news_items):
                sentiment = float(scores[i][1] - scores[i][0])
                results.append(
                    {
                        **item,
                        "sentiment": sentiment,
                        "impact_score": (
                            self._calculate_impact(item, sentiment)
                            if hasattr(self, "_calculate_impact")
                            else 1.0
                        ),
                    }
                )
            mean_sent = np.mean([res["sentiment"] for res in results]) if results else 0
            print(
                f"[SENTIMENT DEBUG] Moyenne batch: {mean_sent:.4f} sur {len(results)} news"
            )
            for i, res in enumerate(results[:5]):
                print(
                    f"  - Sentiment {i+1}: {res['sentiment']:.4f} | Titre: {res['title'][:60]}"
                )
            return results

        except Exception as e:
            print("[DEBUG] EXCEPTION analyze_sentiment_batch:", e)
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return []

    async def update_analysis(self):
        try:
            # 1. Récupération des news
            raw_news = await self.fetch_all_news()
            print(f"[DEBUG update_analysis] Fetched {len(raw_news)} news.")

            # 2. Analyse de sentiment (obligatoire AVANT le buffer)
            analyzed_news = self.analyze_sentiment_batch(raw_news)
            print(
                f"[DEBUG update_analysis] Sentiment analyzed for {len(analyzed_news)} news."
            )

            # 3. Affiche les sentiments trouvés
            for n in analyzed_news[:10]:
                print(
                    f"[DEBUG SENTIMENT BATCH] {n.get('title', '')[:40]} | {n.get('symbols', [])} | sentiment={n.get('sentiment', 'NA')}"
                )

            # 4. Mise à jour du buffer (garder les 200 plus récentes)
            self.news_buffer = [
                *self.news_buffer[-100:],  # Garde les 100 précédentes
                *analyzed_news,
            ][
                -200:
            ]  # Limite totale à 200

            # 5. Sauvegarde de l'état
            await self._save_state()

            return analyzed_news

        except Exception as e:
            self.logger.error(f"Error in news update: {str(e)}")
            return []

    async def get_symbol_sentiment(self, symbol: str) -> float:
        """
        Calcule un score de sentiment pondéré et décayé pour un symbole donné.
        - symbol: string comme 'BTCUSDT' ou 'ETH/USDT'
        - Utilise les news du buffer, pondère par impact_score et applique un decay temporel.
        - Fallback sur le sous-symbole (ex: 'BTC' pour 'BTCUSDT') si rien ne matche.
        """
        try:
            symbol_key = symbol.replace("/", "").upper()  # ex: BTCUSDT
            underlying = symbol_key.replace("USDT", "").replace("USD", "")
            total = 0.0
            total_weight = 0.0
            current_time = datetime.now().timestamp()
            matched = False

            for news in self.news_buffer:
                news_symbols = news.get("symbols", [])
                if symbol_key in news_symbols or underlying in news_symbols:
                    matched = True
                    hours_old = (
                        current_time - news.get("timestamp", current_time)
                    ) / 3600
                    decay = 0.5 ** (hours_old / 24)  # half-life 24h
                    sentiment = news.get("sentiment", 0)
                    impact = news.get("impact_score", 1) or 1
                    total += sentiment * impact * decay
                    total_weight += impact * decay

            if not matched:
                print(
                    f"[DEBUG SENTIMENT] Aucun match trouvé pour {symbol_key} ni {underlying}"
                )

            score = total / max(total_weight, 1e-6) if total_weight > 0 else 0.0
            print(
                f"[DEBUG SENTIMENT] symbol={symbol_key}/{underlying} | sentiment_score={score:.4f} | total={total:.4f} | total_weight={total_weight:.4f}"
            )
            return score

        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return 0.0

    async def _save_state(self, path: Optional[str] = None):
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
                        "symbol_mapping": self.symbol_extractor.symbol_mapping,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")

    def _parse_json(self, data, source=None):
        try:
            if isinstance(data, dict) and "Data" in data:
                news_items = data["Data"]
            elif isinstance(data, dict):
                news_items = data.get("data", []) or data.get("news", []) or []
            elif isinstance(data, str):
                data = json.loads(data)
                news_items = data.get("Data", []) or data.get("data", []) or []
            else:
                return []

            formatted = []
            for item in news_items:
                title = item.get("title", "")
                text = item.get("body", "") or item.get("text", "")
                # Extraction avancée des symboles (titre + texte)
                symbols = self.symbol_extractor.extract_symbols(f"{title} {text}")
                formatted.append(
                    {
                        "title": title,
                        "text": text,
                        "source": source["name"] if source else "Unknown",
                        "timestamp": self._parse_timestamp(item),
                        "url": item.get("url", ""),
                        "symbols": symbols,
                        "source_weight": (
                            source["weight"] if source and "weight" in source else 1.0
                        ),
                    }
                )
            return formatted
        except Exception as e:
            self.logger.error(f"Error parsing JSON news: {e}")
            return []

    def _parse_timestamp(self, item):
        try:
            if isinstance(item, dict):
                if "published_on" in item:
                    return int(item["published_on"])
                if "pubDate" in item:
                    from email.utils import parsedate_to_datetime

                    return int(parsedate_to_datetime(item["pubDate"]).timestamp())
            else:
                pub_date = item.find("pubDate")
                if pub_date:
                    from email.utils import parsedate_to_datetime

                    return int(parsedate_to_datetime(pub_date.text).timestamp())
            return int(datetime.now().timestamp())
        except Exception:
            return int(datetime.now().timestamp())

    def _calculate_impact(self, news: Dict, sentiment: float) -> float:
        """
        Calcule un score d'impact pour la pondération du sentiment.
        Peut être ajusté selon la source, la fraîcheur, la longueur, etc.
        """
        impact = news.get("source_weight", 1.0)
        # Bonus ou malus selon le score de sentiment
        impact *= 1.0 + min(1.0, abs(sentiment))
        # Bonus fraîcheur (moins de 24h = 1.0, sinon decay)
        hours_old = (
            datetime.now().timestamp()
            - news.get("timestamp", datetime.now().timestamp())
        ) / 3600
        impact *= max(0.1, 1.0 - (hours_old / 48))
        # Bonus nombre de symboles
        n_symbols = len(news.get("symbols", []))
        impact *= min(2.0, 1.0 + n_symbols * 0.1)
        return max(0.1, min(5.0, impact))

    # Pour compatibilité legacy :
    def _extract_symbols(self, text: str) -> List[str]:
        return self.symbol_extractor.extract_symbols(text)
