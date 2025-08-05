import os
import re
import json
import logging
import aiohttp
import ssl
import socket
import asyncio
from typing import List, Dict, Optional, Set, Any
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional, Set


class NewsSentimentAnalyzer:
    # Définition du mapping comme attribut de classe
    SYMBOL_MAPPING = {
        "bitcoin": "BTC",
        "btc": "BTC",
        "ethereum": "ETH",
        "eth": "ETH",
        "cardano": "ADA",
        "ada": "ADA",
        "solana": "SOL",
        "sol": "SOL",
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
    }

    def __init__(self, config: dict):
        # API Keys
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.crypto_panic_api_key = os.getenv("CRYPTO_PANIC_API_KEY")
        self.news_api_languages = os.getenv("NEWS_API_LANGUAGES", "en,fr")
        self.news_sources = os.getenv("NEWS_SOURCES", "bloomberg,reuters,coindesk")

        # Sources de news
        self.sources = [
            {
                "name": "CryptoCompare",
                "url": "https://min-api.cryptocompare.com/data/v2/news/?lang=FR",
                "type": "json",
                "weight": 0.7,
            },
            {
                "name": "NewsAPI",
                "url": (
                    "https://newsapi.org/v2/everything?"
                    "q=crypto OR bitcoin OR blockchain&"
                    f"language={self.news_api_languages}&"
                    f"sources={self.news_sources}&"
                    f"apiKey={self.news_api_key}"
                ),
                "type": "json",
                "weight": 0.7,
            },
            {
                "name": "Cointelegraph",
                "url": "https://cointelegraph.com/rss",
                "type": "rss",
                "weight": 0.8,
            },
            {
                "name": "Decrypt",
                "url": "https://decrypt.co/feed",
                "type": "rss",
                "weight": 0.8,
            },
            {
                "name": "NewsBTC",
                "url": "https://www.newsbtc.com/feed/",
                "type": "rss",
                "weight": 0.7,
            },
            {
                "name": "TheBlock",
                "url": "https://www.theblock.co/rss.xml",
                "type": "rss",
                "weight": 0.7,
            },
        ]
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Configuration SSL
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        # Configuration des timeouts et retries
        self.conn_timeout = 10
        self.read_timeout = 20
        self.max_retries = 3
        self.retry_delay = 5

        # Headers HTTP
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

        # Initialisation des patterns regex pour l'extraction des symboles
        self.regex_patterns = [
            (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), ticker)
            for name, ticker in self.SYMBOL_MAPPING.items()
        ]
        self.known_tickers = set(self.SYMBOL_MAPPING.values())

        # Configuration existante
        self.low_watermark_ratio = config.get("news", {}).get(
            "low_watermark_ratio", 0.2
        )
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self._model = None
        self._tokenizer = None

        # Buffer et configuration
        self.news_buffer = []
        self.sentiment_weight = config.get("news", {}).get("sentiment_weight", 0.15)
        self.update_interval = config.get("news", {}).get("update_interval", 300)

    def extract_symbols(self, text: str) -> List[str]:
        """
        Extrait les symboles de crypto-monnaies du texte.

        Args:
            text (str): Le texte à analyser

        Returns:
            List[str]: Liste des symboles trouvés
        """
        found: Set[str] = set()
        if not text:
            return []

        # Recherche par regex patterns
        for pattern, ticker in self.regex_patterns:
            if pattern.search(text):
                found.add(ticker)

        # Recherche des paires de trading
        for pair in re.findall(
            r"\b([A-Z]{3,5})[/-]?(USDT|USD|BTC|ETH)?\b", text.upper()
        ):
            ticker = pair[0]
            if ticker in self.known_tickers:
                found.add(ticker)

        # Recherche des symboles avec $
        for match in re.findall(r"\$([A-Z]{3,5})\b", text.upper()):
            if match in self.known_tickers:
                found.add(match)

        return list(found)

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            )
            self._model.to(self.device)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        return self._tokenizer

    def normalize_timestamp(self, timestamp: Any) -> int:
        """Normalise un timestamp en secondes depuis l'époque"""
        try:
            # Si c'est déjà un int
            if isinstance(timestamp, int):
                if timestamp > 9999999999:  # Si en millisecondes
                    return int(timestamp / 1000)
                return timestamp

            # Si c'est une string
            if isinstance(timestamp, str):
                # Essaye de parser comme int d'abord
                try:
                    ts = int(timestamp)
                    if ts > 9999999999:  # Si en millisecondes
                        return int(ts / 1000)
                    return ts
                except ValueError:
                    # Si ce n'est pas un int, essaye de parser comme datetime
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    return int(dt.timestamp())

            # Si c'est un datetime
            if isinstance(timestamp, datetime):
                return int(timestamp.timestamp())

            # Si on ne peut pas convertir, retourne le timestamp actuel
            return int(datetime.now().timestamp())

        except Exception as e:
            self.logger.warning(f"Erreur normalisation timestamp {timestamp}: {str(e)}")
            return int(datetime.now().timestamp())

    async def _save_state(self, data):
        path = self.config.get("news", {}).get(
            "storage_path", "data/news_analysis.json"
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"[NEWS] State saved to {path}")
        except Exception as e:
            self.logger.error(f"[NEWS] Failed to save state: {e}")

    async def fetch_all_news(self) -> List[Dict]:
        """Récupère les news de toutes les sources avec gestion des erreurs améliorée"""
        connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            limit=10,  # Limite de connexions simultanées
            ttl_dns_cache=300,  # Cache DNS de 5 minutes
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=self.read_timeout, connect=self.conn_timeout
        )

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=self.headers
        ) as session:
            tasks = [
                self._fetch_source_with_retry(session, source)
                for source in self.sources
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_news = []

            for source, result in zip(self.sources, results):
                if isinstance(result, Exception):
                    self.logger.error(f"[{source['name']}] Échec: {str(result)}")
                    continue
                if isinstance(result, list):
                    valid_news.extend(result)
                    if result:  # Log des premières news récupérées
                        self.logger.info(
                            f"[{source['name']}] {len(result)} news récupérées"
                        )
                        for n in result[:2]:  # Log des 2 premiers titres
                            self.logger.info(f"  - {n.get('title', '')[:100]}")

            # Patch et mise à jour du buffer
            valid_news = self.patch_news_list(valid_news)
            self.news_buffer = valid_news

            return valid_news

    async def _fetch_source_with_retry(
        self, session: aiohttp.ClientSession, source: Dict, max_retries=None
    ) -> List[Dict]:
        max_retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                # Calcul du délai exponentiel
                delay = self.retry_delay * (2**attempt) + (random.random() * 2)

                if attempt > 0:
                    self.logger.info(
                        f"[{source['name']}] Tentative {attempt+1}/{max_retries} après {delay:.1f}s"
                    )

                # Configuration du timeout pour cette tentative
                timeout = aiohttp.ClientTimeout(
                    total=self.read_timeout * (attempt + 1), connect=self.conn_timeout
                )

                async with session.get(
                    source["url"],
                    timeout=timeout,
                    ssl=self.ssl_context,
                    headers=self.headers,
                ) as response:
                    if response.status == 429:  # Too Many Requests
                        retry_after = int(response.headers.get("Retry-After", delay))
                        self.logger.warning(
                            f"[{source['name']}] Rate limit atteint, attente de {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status != 200:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                        )

                    content = await response.text()

                    # Parse le contenu selon le type
                    if source["type"] == "rss":
                        news = self._parse_rss(content, source)
                    else:
                        data = json.loads(content)
                        news = self._parse_json(data, source)

                    if news:  # Si on a des résultats valides
                        return news

                    # Si pas de news et ce n'est pas le dernier essai
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        continue

                    return []

            except asyncio.TimeoutError as e:
                last_error = f"Timeout après {self.read_timeout * (attempt + 1)}s"
                self.logger.warning(f"[{source['name']}] {last_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    continue

            except aiohttp.ClientError as e:
                last_error = f"Erreur réseau: {str(e)}"
                self.logger.warning(f"[{source['name']}] {last_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                last_error = f"Erreur inattendue: {str(e)}"
                self.logger.error(f"[{source['name']}] {last_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    continue

        self.logger.error(
            f"[{source['name']}] Échec après {max_retries} tentatives: {last_error}"
        )
        return []

    async def _fetch_source(
        self, session: aiohttp.ClientSession, source: Dict
    ) -> List[Dict]:
        try:
            async with session.get(source["url"], timeout=30) as response:
                body = await response.text()
                if response.status == 429:
                    self.logger.error(
                        f"[{source['name']}] HTTP 429 Too Many Requests ({source['url']}) | Body: {body}"
                    )
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message="Too Many Requests",
                        headers=response.headers,
                    )
                if response.status != 200:
                    self.logger.error(
                        f"[{source['name']}] HTTP status {response.status} ({source['url']}) | Body: {body}"
                    )
                    return []
                if source["type"] == "rss":
                    return self._parse_rss(body, source)
                else:
                    try:
                        data = json.loads(body)
                    except Exception as e:
                        self.logger.error(
                            f"[{source['name']}] Failed to parse JSON: {str(e)} | Body: {body}"
                        )
                        return []
                    return self._parse_json(data, source)
        except asyncio.TimeoutError:
            self.logger.error(
                f"[{source['name']}] Timeout when fetching ({source['url']})"
            )
            return []
        except Exception as e:
            self.logger.error(
                f"[{source['name']}] Error fetching: {str(e)} ({source['url']})"
            )
            return []

    def _parse_rss(self, content: str, source: Dict) -> List[Dict]:
        try:
            soup = BeautifulSoup(content, "xml")
            items = soup.find_all("item")
            return [self._parse_rss_item(item, source) for item in items]
        except Exception as e:
            self.logger.error(f"Error parsing RSS {source['name']}: {str(e)}")
            return []

    def _parse_json(self, data, source: Dict) -> List[Dict]:
        news_list = []
        if source["name"] == "CryptoCompare" and "Data" in data:
            for n in data["Data"]:
                title = n.get("title", "")
                text = n.get("body", "")
                url = n.get("url", "")
                symbols = self.extract_symbols(f"{title} {text}")

                # Utilise normalize_timestamp
                timestamp = self.normalize_timestamp(n.get("published_on", None))

                news_list.append(
                    {
                        "title": title,
                        "text": text,
                        "source": source["name"],
                        "timestamp": timestamp,
                        "url": url,
                        "symbols": symbols,
                        "source_weight": source["weight"],
                    }
                )

        elif source["name"] == "NewsAPI" and "articles" in data:
            for n in data["articles"]:
                title = n.get("title", "")
                text = n.get("description", "") or n.get("content", "")
                url = n.get("url", "")
                symbols = self.extract_symbols(f"{title} {text}")

                # Utilise normalize_timestamp
                timestamp = self.normalize_timestamp(n.get("publishedAt", None))

                news_list.append(
                    {
                        "title": title,
                        "text": text,
                        "source": source["name"],
                        "timestamp": timestamp,
                        "url": url,
                        "symbols": symbols,
                        "source_weight": source["weight"],
                    }
                )
        return news_list

    def _parse_rss_item(self, item, source: Dict) -> Dict:
        title = item.find("title").text if item.find("title") else ""
        description = item.find("description").text if item.find("description") else ""
        url = item.find("link").text if item.find("link") else ""
        symbols = self.extract_symbols(f"{title} {description}")

        # Extraction et normalisation du timestamp
        pub_date = item.find("pubDate")
        timestamp = self.normalize_timestamp(pub_date.text if pub_date else None)

        return {
            "title": title,
            "text": description,
            "source": source["name"],
            "timestamp": timestamp,
            "url": url,
            "symbols": symbols,
            "source_weight": source["weight"],
        }

    def _parse_timestamp(self, item):
        pub_date = item.find("pubDate")
        if pub_date and pub_date.text:
            try:
                # Convertir directement en timestamp
                dt = datetime.strptime(pub_date.text[:25], "%a, %d %b %Y %H:%M:%S")
                return int(dt.timestamp())
            except Exception as e:
                self.logger.warning(f"Erreur parsing date {pub_date.text}: {e}")
        return int(datetime.now().timestamp())

    def analyze_sentiment_batch(
        self, news_items: List[Dict], low_watermark_ratio: float = None
    ) -> List[Dict]:
        news_items = self.patch_news_list(news_items)
        if low_watermark_ratio is None:
            low_watermark_ratio = self.low_watermark_ratio
        try:
            low_watermark_ratio = float(low_watermark_ratio)
        except Exception:
            low_watermark_ratio = 0.2
        if low_watermark_ratio > 0.5 or low_watermark_ratio < 0.05:
            print(
                f"[DEBUG] Watermark ratio {low_watermark_ratio} is invalid, forcing to 0.2"
            )
            low_watermark_ratio = 0.2
        if not news_items:
            print("[SENTIMENT] Aucun article à analyser.")
            return []
        try:
            texts = [f"{item['title']}. {item['text']}"[:512] for item in news_items]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            results = []
            for i, item in enumerate(news_items):
                sentiment = float(scores[i][2] - scores[i][0])
                results.append({**item, "sentiment": sentiment, "impact_score": 1.0})
            return results
        except Exception as e:
            print("[DEBUG] EXCEPTION analyze_sentiment_batch:", e)
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return []

    async def update_analysis(self):
        try:
            raw_news = await self.fetch_all_news()
            self.logger.debug(f"Fetched {len(raw_news)} raw news items")
            analyzed_news = self.analyze_sentiment_batch(raw_news)
            if not isinstance(analyzed_news, list):
                self.logger.error(
                    f"Invalid sentiment results type: {type(analyzed_news)}"
                )
                analyzed_news = []
            sentiment_scores = [
                n.get("sentiment", 0) for n in analyzed_news if isinstance(n, dict)
            ]
            mean_sentiment = (
                float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
            )
            std_sentiment = float(np.std(sentiment_scores)) if sentiment_scores else 0.0
            self.logger.info(
                f"News analysis: {len(analyzed_news)} items | Mean sentiment: {mean_sentiment:.4f} ± {std_sentiment:.4f}"
            )
            self.news_buffer = analyzed_news[-200:]
            summary = self.get_sentiment_summary()
            await self._save_state(
                {
                    "mean_sentiment": mean_sentiment,
                    "std_sentiment": std_sentiment,
                    "analyzed_news": analyzed_news[:50],
                    "sentiment_global": summary.get("sentiment_global", 0.0),
                    "top_symbols": summary.get("top_symbols", []),
                    "top_news": summary.get("top_news", []),
                }
            )
            return {
                "mean": mean_sentiment,
                "std": std_sentiment,
                "scores": sentiment_scores,
                "items": analyzed_news,
            }
        except Exception as e:
            self.logger.error(f"News update failed: {str(e)}", exc_info=True)
            return {"mean": 0.0, "std": 0.0, "scores": [], "items": []}

    def get_sentiment_summary(self, top_n=5):
        valid = [
            item
            for item in self.news_buffer
            if "sentiment" in item and item["sentiment"] is not None
        ]
        if not valid:
            return {
                "sentiment_global": 0.0,
                "n_news": 0,
                "top_symbols": [],
                "top_news": [],
            }
        sentiments = [item["sentiment"] for item in valid]
        sentiment_global = float(np.mean(sentiments))
        top_news = sorted(valid, key=lambda x: abs(x["sentiment"]), reverse=True)[
            :top_n
        ]
        top_news_titles = [news["title"] for news in top_news if "title" in news]
        symbol_scores = {}
        for item in valid:
            for s in item.get("symbols", []):
                symbol_scores.setdefault(s, []).append(item["sentiment"])
        top_symbols = sorted(
            symbol_scores.items(), key=lambda kv: abs(np.mean(kv[1])), reverse=True
        )
        top_symbols = [s for s, scores in top_symbols[:top_n]]
        return {
            "sentiment_global": sentiment_global,
            "n_news": len(valid),
            "top_symbols": top_symbols,
            "top_news": top_news_titles,
        }

    async def get_symbol_sentiment(
        self, symbol: str, news_list: Optional[list] = None
    ) -> float:
        """
        Calcule le sentiment moyen pour un symbole donné avec decay temporel.

        Args:
            symbol (str): Le symbole de la crypto-monnaie (ex: "BTC", "ETH")
            news_list (Optional[list]): Liste optionnelle de news, utilise news_buffer si None

        Returns:
            float: Score de sentiment entre -1 et 1
        """
        try:
            # Normalisation du symbole
            symbol_key = symbol.replace("/", "").upper()

            # Mapping des symboles vers leurs variations
            coin_mapping = {
                "BTC": ["BTC", "BITCOIN", "XBT"],
                "ETH": ["ETH", "ETHEREUM", "ETHER"],
                "SOL": ["SOL", "SOLANA"],
                "ADA": ["ADA", "CARDANO"],
                "TRX": ["TRX", "TRON"],
                "BNB": ["BNB", "BINANCE", "BINANCECOIN"],
                "XRP": ["XRP", "RIPPLE"],
                "DOGE": ["DOGE", "DOGECOIN"],
                "AVAX": ["AVAX", "AVALANCHE"],
                "DOT": ["DOT", "POLKADOT"],
                "MATIC": ["MATIC", "POLYGON"],
                "LINK": ["LINK", "CHAINLINK"],
                "UNI": ["UNI", "UNISWAP"],
                "AAVE": ["AAVE"],
                "ATOM": ["ATOM", "COSMOS"],
                "NEAR": ["NEAR", "NEAR PROTOCOL"],
                "ALGO": ["ALGO", "ALGORAND"],
                "FTM": ["FTM", "FANTOM"],
                "XLM": ["XLM", "STELLAR"],
                "HBAR": ["HBAR", "HEDERA"],
            }

            # Identification du coin
            coin = None
            for cm in sorted(coin_mapping.keys(), key=len, reverse=True):
                if symbol_key.startswith(cm):
                    coin = cm
                    break

            # Fallback sur les 3 premiers caractères si non trouvé
            if coin is None:
                coin = symbol_key[:3]

            # Récupération des termes de recherche
            search_terms = coin_mapping.get(coin, [coin])

            # Utilisation du buffer si pas de liste fournie
            if news_list is None:
                news_list = self.news_buffer

            # Variables d'accumulation
            total_sentiment = 0.0
            total_weight = 0.0
            matched_news = 0

            # Timestamp actuel en UTC
            current_time = datetime.now(timezone.utc).timestamp()

            # Parcours des news
            for news in news_list:
                try:
                    # Extraction des données de la news
                    news_symbols = news.get("symbols", [])
                    title = news.get("title", "").lower()
                    text = news.get("text", "").lower()
                    content = f"{title} {text}"
                    timestamp = self.normalize_timestamp(
                        news.get("timestamp", current_time)
                    )

                    # Vérification des correspondances
                    match_extracted = any(
                        s.upper().strip() in [term.upper() for term in search_terms]
                        for s in news_symbols
                    )
                    match_content = any(
                        term.lower() in content for term in search_terms
                    )

                    # Si correspondance trouvée
                    if match_extracted or match_content:
                        # Calcul de l'âge de la news en heures
                        hours_old = (current_time - timestamp) / 3600

                        # Facteur de decay exponentiel (diminue de moitié toutes les 24h)
                        decay = 0.5 ** (hours_old / 24)

                        # Récupération du sentiment et de l'impact
                        sentiment = float(news.get("sentiment", 0))
                        impact = float(news.get("impact_score", 1) or 1)

                        # Source weight pour pondération
                        source_weight = float(news.get("source_weight", 0.7))

                        # Calcul du poids final
                        weight = decay * impact * source_weight

                        # Accumulation
                        total_sentiment += sentiment * weight
                        total_weight += weight
                        matched_news += 1

                except Exception as e:
                    self.logger.warning(
                        f"Erreur traitement news pour {symbol}: {str(e)}"
                    )
                    continue

            # Calcul du score final
            if total_weight > 0:
                final_score = total_sentiment / total_weight
                # Log du résultat
                self.logger.info(
                    f"Sentiment {symbol}: {final_score:.3f} (basé sur {matched_news} news)"
                )
                return float(max(min(final_score, 1.0), -1.0))  # Clamp entre -1 et 1
            else:
                self.logger.info(f"Pas de sentiment calculable pour {symbol}")
                return 0.0

        except Exception as e:
            self.logger.error(f"Erreur calcul sentiment pour {symbol}: {str(e)}")
            return 0.0

    def _extract_symbols_from_text(self, text):
        text = text.lower()
        found = set()
        for key, symbol in self.SYMBOL_MAPPING.items():
            if re.search(r"\b" + re.escape(key) + r"\b", text):
                found.add(symbol)
        return list(found)

    def patch_news_item(self, news):
        if "symbols" not in news or not news["symbols"]:
            title = news.get("title", "") or ""
            text = news.get("text", "") or ""
            symbols = self._extract_symbols_from_text(title + " " + text)
            news["symbols"] = symbols
        if "sentiment" not in news or news["sentiment"] is None:
            news["sentiment"] = 0.0
        else:
            try:
                news["sentiment"] = float(news["sentiment"])
            except Exception:
                news["sentiment"] = 0.0
        return news

    def patch_news_list(self, news_list):
        return [self.patch_news_item(news) for news in news_list]
