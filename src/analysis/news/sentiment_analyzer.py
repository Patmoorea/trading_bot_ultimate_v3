import os
import json
import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import logging
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import aiohttp
import ssl
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


@dataclass
class NewsItem:
    """Data class for news items with sentiment analysis."""

    title: str
    text: str
    source: str
    timestamp: float
    url: str
    symbols: List[str]
    source_weight: float
    sentiment: Optional[float] = None
    impact_score: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class NewsSource:
    """Configuration for news sources."""

    name: str
    url: str
    type: str  # 'rss' or 'json'
    weight: float
    enabled: bool = True
    timeout: int = 30


class SymbolExtractor:
    """Enhanced symbol extraction with comprehensive cryptocurrency patterns."""

    def __init__(self):
        self.symbol_patterns = {
            r"\b(?:BTC|BITCOIN)\b": "BTC",
            r"\b(?:ETH|ETHEREUM)\b": "ETH",
            r"\b(?:SOL|SOLANA)\b": "SOL",
            r"\b(?:ADA|CARDANO)\b": "ADA",
            r"\b(?:DOT|POLKADOT)\b": "DOT",
            r"\b(?:AVAX|AVALANCHE)\b": "AVAX",
            r"\b(?:MATIC|POLYGON)\b": "MATIC",
            r"\b(?:LINK|CHAINLINK)\b": "LINK",
            r"\b(?:UNI|UNISWAP)\b": "UNI",
            r"\b(?:ATOM|COSMOS)\b": "ATOM",
            r"\b(?:ALGO|ALGORAND)\b": "ALGO",
            r"\b(?:XRP|RIPPLE)\b": "XRP",
            r"\b(?:LTC|LITECOIN)\b": "LTC",
            r"\b(?:BCH|BITCOIN CASH)\b": "BCH",
            r"\b(?:XLM|STELLAR)\b": "XLM",
            r"\b(?:DOGE|DOGECOIN)\b": "DOGE",
            r"\b(?:SHIB|SHIBA INU)\b": "SHIB",
            r"\b(?:NEAR|NEAR PROTOCOL)\b": "NEAR",
            r"\b(?:FTM|FANTOM)\b": "FTM",
            r"\b(?:SAND|SANDBOX)\b": "SAND",
            r"\b(?:MANA|DECENTRALAND)\b": "MANA",
            r"\b(?:APE|APECOIN)\b": "APE",
            r"\b(?:CRO|CRONOS)\b": "CRO",
            r"\b(?:HBAR|HEDERA)\b": "HBAR",
            r"\b(?:VET|VECHAIN)\b": "VET",
            r"\b(?:ICP|INTERNET COMPUTER)\b": "ICP",
            r"\b(?:THETA|THETA NETWORK)\b": "THETA",
            r"\b(?:FIL|FILECOIN)\b": "FIL",
            r"\b(?:TRX|TRON)\b": "TRX",
            r"\b(?:ETC|ETHEREUM CLASSIC)\b": "ETC",
            r"\b(?:XMR|MONERO)\b": "XMR",
            r"\b(?:AAVE)\b": "AAVE",
            r"\b(?:MKR|MAKER)\b": "MKR",
            r"\b(?:COMP|COMPOUND)\b": "COMP",
            r"\b(?:SUSHI|SUSHISWAP)\b": "SUSHI",
            r"\b(?:YFI|YEARN FINANCE)\b": "YFI",
            r"\b(?:SNX|SYNTHETIX)\b": "SNX",
            r"\b(?:CRV|CURVE)\b": "CRV",
            r"\b(?:BAL|BALANCER)\b": "BAL",
            r"\b(?:ZRX|0X)\b": "ZRX",
            r"\b(?:BAT|BASIC ATTENTION TOKEN)\b": "BAT",
            r"\b(?:ENJ|ENJIN)\b": "ENJ",
            r"\b(?:CHZ|CHILIZ)\b": "CHZ",
            r"\b(?:HOT|HOLO)\b": "HOT",
            r"\b(?:ZIL|ZILLIQA)\b": "ZIL",
            r"\b(?:QTUM)\b": "QTUM",
            r"\b(?:ONT|ONTOLOGY)\b": "ONT",
            r"\b(?:ZEC|ZCASH)\b": "ZEC",
            r"\b(?:DASH)\b": "DASH",
            r"\b(?:DCR|DECRED)\b": "DCR",
            r"\b(?:DGB|DIGIBYTE)\b": "DGB",
            r"\b(?:RVN|RAVENCOIN)\b": "RVN",
            r"\b(?:WAVES)\b": "WAVES",
            r"\b(?:KSM|KUSAMA)\b": "KSM",
            r"\b(?:FLOW)\b": "FLOW",
            r"\b(?:EGLD|ELROND)\b": "EGLD",
            r"\b(?:ONE|HARMONY)\b": "ONE",
            r"\b(?:CELO)\b": "CELO",
            r"\b(?:AR|ARWEAVE)\b": "AR",
            r"\b(?:GRT|GRAPH)\b": "GRT",
            r"\b(?:LRC|LOOPRING)\b": "LRC",
            r"\b(?:ENS|ETHEREUM NAME SERVICE)\b": "ENS",
            r"\b(?:IMX|IMMUTABLE X)\b": "IMX",
            r"\b(?:GALA)\b": "GALA",
            r"\b(?:AXS|AXIE INFINITY)\b": "AXS",
            r"\b(?:SLP|SMOOTH LOVE POTION)\b": "SLP",
        }
        self.compiled_patterns = {
            re.compile(pattern, re.IGNORECASE): symbol
            for pattern, symbol in self.symbol_patterns.items()
        }

    def extract_symbols(self, text: str) -> List[str]:
        if not text:
            return []
        symbols = set()
        text_upper = text.upper()
        for pattern, symbol in self.compiled_patterns.items():
            if pattern.search(text):
                symbols.add(symbol)
        symbols.update(self._extract_contextual_symbols(text_upper))
        return list(symbols)

    def _extract_contextual_symbols(self, text: str) -> Set[str]:
        symbols = set()
        trading_pairs = re.findall(r"\b([A-Z]{2,5})[-/]?USDT?\b", text)
        for pair in trading_pairs:
            if pair in ["USDT", "USD", "EUR", "GBP"]:
                continue
            symbols.add(pair)
        price_patterns = re.findall(r"\$([A-Z]{2,5})\b", text)
        symbols.update(price_patterns)
        return symbols


class NewsSentimentAnalyzer:
    """Advanced news sentiment analyzer for cryptocurrency trading."""

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.symbol_extractor = SymbolExtractor()
        self._model = None
        self._tokenizer = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.news_sources = self._initialize_sources()
        self.symbol_mapping = config.get("symbol_mapping", {})
        self.news_buffer: List[NewsItem] = []
        self.max_buffer_size = config.get("news", {}).get("max_buffer_size", 500)
        self.sentiment_weight = config.get("news", {}).get("sentiment_weight", 0.15)
        self.update_interval = config.get("news", {}).get("update_interval", 300)
        self.min_confidence_threshold = config.get("news", {}).get(
            "min_confidence", 0.6
        )
        self.batch_size = config.get("news", {}).get("batch_size", 16)
        self.max_text_length = config.get("news", {}).get("max_text_length", 512)
        self._sentiment_cache = {}
        self._cache_ttl = 300

    async def _save_state(self, path: Optional[str] = None):
        """
        Sauvegarde l'état courant des news (buffer, config, etc).
        """
        try:
            path = path or self.config.get("news", {}).get(
                "storage_path", "data/news_analysis.json"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {
                        "last_updated": datetime.now().isoformat(),
                        "news_count": len(self.news_buffer),
                        "symbol_mapping": self.symbol_mapping,
                        # Tu peux aussi ajouter le sentiment global, top_symbols, etc
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")

    def _initialize_sources(self) -> List[NewsSource]:
        """Initialize news sources from configuration."""
        default_sources = [
            NewsSource(
                name="CoinDesk",
                url="https://www.coindesk.com/arc/outboundfeeds/rss/",
                type="rss",
                weight=0.9,
            ),
            NewsSource(
                name="CryptoCompare",
                url="https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
                type="json",
                weight=0.7,
            ),
            NewsSource(
                name="Cointelegraph",
                url="https://cointelegraph.com/rss",
                type="rss",
                weight=0.8,
            ),
            NewsSource(
                name="CoinTelegraph Bitcoin",
                url="https://cointelegraph.com/rss/tag/bitcoin",
                type="rss",
                weight=0.85,
            ),
            NewsSource(
                name="CryptoNews",
                url="https://cryptonews.com/news/feed/",
                type="rss",
                weight=0.75,
            ),
            NewsSource(
                name="Decrypt", url="https://decrypt.co/feed", type="rss", weight=0.8
            ),
        ]

        # Override with config if provided
        config_sources = self.config.get("news", {}).get("sources", [])
        if config_sources:
            return [NewsSource(**source) for source in config_sources]

        return default_sources

    @property
    def model(self):
        """Lazy loading of the sentiment analysis model."""
        if self._model is None:
            try:
                model_name = self.config.get("news", {}).get(
                    "model_name", "ProsusAI/finbert"
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self._model.to(self._device)
                self._model.eval()
                self.logger.info(
                    f"Loaded sentiment model: {model_name} on {self._device}"
                )
            except Exception as e:
                self.logger.error(f"Failed to load sentiment model: {e}")
                raise
        return self._model

    @property
    def tokenizer(self):
        """Lazy loading of the tokenizer."""
        if self._tokenizer is None:
            try:
                model_name = self.config.get("news", {}).get(
                    "model_name", "ProsusAI/finbert"
                )
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.logger.info(f"Loaded tokenizer: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load tokenizer: {e}")
                raise
        return self._tokenizer

    async def fetch_all_news(self) -> List[NewsItem]:
        """Fetch news from all configured sources."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context, limit=10),
            headers=headers,
            timeout=timeout,
        ) as session:
            # Filter enabled sources
            active_sources = [source for source in self.news_sources if source.enabled]

            # Create tasks for concurrent fetching
            tasks = [
                self._fetch_source_with_retry(session, source)
                for source in active_sources
            ]

            # Execute with timeout and error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            all_news = []
            for i, result in enumerate(results):
                source_name = active_sources[i].name
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching {source_name}: {result}")
                elif isinstance(result, list):
                    all_news.extend(result)
                    self.logger.debug(
                        f"Fetched {len(result)} articles from {source_name}"
                    )

            # Enhance symbols and deduplicate
            enhanced_news = self._enhance_and_deduplicate(all_news)

            self.logger.info(
                f"Successfully fetched {len(enhanced_news)} unique news articles"
            )
            return enhanced_news

    async def _fetch_source_with_retry(
        self, session: aiohttp.ClientSession, source: NewsSource, max_retries: int = 3
    ) -> List[NewsItem]:
        """Fetch news from a single source with retry logic."""
        for attempt in range(max_retries):
            try:
                async with session.get(source.url, timeout=source.timeout) as response:
                    if response.status == 200:
                        if source.type == "rss":
                            content = await response.text()
                            return self._parse_rss(content, source)
                        else:
                            data = await response.json()
                            return self._parse_json(data, source)
                    else:
                        self.logger.warning(f"HTTP {response.status} for {source.name}")

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout fetching {source.name} (attempt {attempt + 1})"
                )
            except Exception as e:
                self.logger.error(
                    f"Error fetching {source.name} (attempt {attempt + 1}): {e}"
                )

            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        return []

    def _enhance_and_deduplicate(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Enhance symbol extraction and remove duplicates."""
        seen_urls = set()
        seen_titles = set()
        unique_news = []

        for item in news_items:
            # Skip duplicates based on URL or title similarity
            if item.url in seen_urls:
                continue

            title_key = item.title.lower().strip()[
                :100
            ]  # First 100 chars for similarity
            if title_key in seen_titles:
                continue

            # Enhance symbol extraction
            title_symbols = self.symbol_extractor.extract_symbols(item.title)
            text_symbols = self.symbol_extractor.extract_symbols(item.text)
            combined_symbols = list(set(item.symbols + title_symbols + text_symbols))

            # Create enhanced item
            enhanced_item = NewsItem(
                title=item.title,
                text=item.text,
                source=item.source,
                timestamp=item.timestamp,
                url=item.url,
                symbols=combined_symbols,
                source_weight=item.source_weight,
            )

            unique_news.append(enhanced_item)
            seen_urls.add(item.url)
            seen_titles.add(title_key)

        # Sort by timestamp (newest first)
        unique_news.sort(key=lambda x: x.timestamp, reverse=True)

        return unique_news

    def _parse_rss(self, content: str, source: NewsSource) -> List[NewsItem]:
        """Parse RSS feed content."""
        try:
            soup = BeautifulSoup(content, "xml")
            items = soup.find_all("item")

            news_items = []
            for item in items:
                try:
                    news_item = self._parse_rss_item(item, source)
                    if news_item:
                        news_items.append(news_item)
                except Exception as e:
                    self.logger.debug(f"Error parsing RSS item from {source.name}: {e}")

            return news_items

        except Exception as e:
            self.logger.error(f"Error parsing RSS from {source.name}: {e}")
            return []

    def _parse_rss_item(self, item, source: NewsSource) -> Optional[NewsItem]:
        """Parse individual RSS item."""
        try:
            title_elem = item.find("title")
            desc_elem = item.find("description")
            link_elem = item.find("link")

            title = title_elem.text.strip() if title_elem else ""
            description = desc_elem.text.strip() if desc_elem else ""
            url = link_elem.text.strip() if link_elem else ""

            if not title:
                return None

            # Clean description (remove HTML tags)
            if description:
                description = BeautifulSoup(description, "html.parser").get_text()

            timestamp = self._parse_timestamp(item)
            symbols = self.symbol_extractor.extract_symbols(f"{title} {description}")

            return NewsItem(
                title=title,
                text=description,
                source=source.name,
                timestamp=timestamp,
                url=url,
                symbols=symbols,
                source_weight=source.weight,
            )

        except Exception as e:
            self.logger.debug(f"Error parsing RSS item: {e}")
            return None

    def _parse_json(self, data: Dict, source: NewsSource) -> List[NewsItem]:
        """Parse JSON API response."""
        try:
            # Handle different JSON structures
            if isinstance(data, dict) and "Data" in data:
                news_items = data["Data"]
            elif isinstance(data, dict):
                news_items = (
                    data.get("data", [])
                    or data.get("news", [])
                    or data.get("articles", [])
                )
            else:
                return []

            parsed_items = []
            for item in news_items:
                try:
                    news_item = self._parse_json_item(item, source)
                    if news_item:
                        parsed_items.append(news_item)
                except Exception as e:
                    self.logger.debug(
                        f"Error parsing JSON item from {source.name}: {e}"
                    )

            return parsed_items

        except Exception as e:
            self.logger.error(f"Error parsing JSON from {source.name}: {e}")
            return []

    def _parse_json_item(self, item: Dict, source: NewsSource) -> Optional[NewsItem]:
        """Parse individual JSON news item."""
        try:
            title = item.get("title", "").strip()
            text = (
                item.get("body", "")
                or item.get("text", "")
                or item.get("description", "")
            )
            url = item.get("url", "") or item.get("guid", "")

            if not title:
                return None

            timestamp = self._parse_timestamp(item)
            symbols = self.symbol_extractor.extract_symbols(f"{title} {text}")

            return NewsItem(
                title=title,
                text=text.strip(),
                source=source.name,
                timestamp=timestamp,
                url=url,
                symbols=symbols,
                source_weight=source.weight,
            )

        except Exception as e:
            self.logger.debug(f"Error parsing JSON item: {e}")
            return None

    def _parse_timestamp(self, item) -> float:
        """Parse timestamp from various formats."""
        try:
            # Handle different timestamp formats
            if hasattr(item, "get"):  # JSON item
                if "published_on" in item:
                    return float(item["published_on"])
                elif "publishedAt" in item:
                    return datetime.fromisoformat(
                        item["publishedAt"].replace("Z", "+00:00")
                    ).timestamp()
                elif "pubDate" in item:
                    return parsedate_to_datetime(item["pubDate"]).timestamp()
            else:  # RSS item
                pub_date = item.find("pubDate")
                if pub_date:
                    return parsedate_to_datetime(pub_date.text).timestamp()

                # Try other date fields
                for date_field in ["dc:date", "published", "updated"]:
                    date_elem = item.find(date_field)
                    if date_elem:
                        return datetime.fromisoformat(
                            date_elem.text.replace("Z", "+00:00")
                        ).timestamp()

            return datetime.now().timestamp()

        except Exception as e:
            self.logger.debug(f"Error parsing timestamp: {e}")
            return datetime.now().timestamp()

    def analyze_sentiment_batch(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Analyze sentiment for a batch of news items with enhanced processing."""
        if not news_items:
            self.logger.debug("No news items to analyze")
            return []

        try:
            # Prepare texts for analysis
            texts = []
            for item in news_items:
                combined_text = f"{item.title}. {item.text}"
                # Truncate to max length
                if len(combined_text) > self.max_text_length:
                    combined_text = combined_text[: self.max_text_length]
                texts.append(combined_text)

            # Process in batches to avoid memory issues
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                batch_items = news_items[i : i + self.batch_size]
                batch_results = self._process_sentiment_batch(batch_texts, batch_items)
                results.extend(batch_results)

            # Calculate statistics
            sentiments = [
                item.sentiment for item in results if item.sentiment is not None
            ]
            if sentiments:
                mean_sentiment = np.mean(sentiments)
                std_sentiment = np.std(sentiments)
                self.logger.info(
                    f"Sentiment analysis complete: {len(results)} items, "
                    f"mean={mean_sentiment:.4f}, std={std_sentiment:.4f}"
                )

            return results

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return news_items  # Return original items without sentiment

    def _process_sentiment_batch(
        self, texts: List[str], items: List[NewsItem]
    ) -> List[NewsItem]:
        """Process a single batch for sentiment analysis."""
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
            )

            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Process results
            results = []
            for i, item in enumerate(items):
                probs = probabilities[i].cpu().numpy()

                # Calculate sentiment score (positive - negative)
                sentiment_score = float(
                    probs[1] - probs[0]
                )  # Assuming [negative, positive, neutral]
                confidence = float(np.max(probs))

                # Calculate impact score
                impact_score = self._calculate_impact_score(
                    item, sentiment_score, confidence
                )

                # Create new item with sentiment data
                enhanced_item = NewsItem(
                    title=item.title,
                    text=item.text,
                    source=item.source,
                    timestamp=item.timestamp,
                    url=item.url,
                    symbols=item.symbols,
                    source_weight=item.source_weight,
                    sentiment=sentiment_score,
                    impact_score=impact_score,
                    confidence=confidence,
                )

                results.append(enhanced_item)

            return results

        except Exception as e:
            self.logger.error(f"Error processing sentiment batch: {e}")
            return items

    def _calculate_impact_score(
        self, item: NewsItem, sentiment: float, confidence: float
    ) -> float:
        """Calculate impact score based on various factors."""
        try:
            # Base impact from source weight
            impact = item.source_weight

            # Adjust for sentiment magnitude
            impact *= 1 + abs(sentiment)

            # Adjust for confidence
            impact *= confidence

            # Adjust for recency (more recent = higher impact)
            hours_old = (datetime.now().timestamp() - item.timestamp) / 3600
            recency_factor = max(0.1, 1.0 - (hours_old / 48))  # Decay over 48 hours
            impact *= recency_factor

            # Adjust for symbol count (more symbols = potentially higher impact)
            symbol_factor = min(2.0, 1.0 + len(item.symbols) * 0.1)
            impact *= symbol_factor

            # Normalize to reasonable range
            return min(5.0, max(0.1, impact))

        except Exception as e:
            self.logger.debug(f"Error calculating impact score: {e}")
            return 1.0

    async def update_analysis(self) -> List[NewsItem]:
        """Update news analysis with new data."""
        try:
            self.logger.info("Starting news analysis update")

            # Fetch new news
            raw_news = await self.fetch_all_news()
            self.logger.info(f"Fetched {len(raw_news)} raw news items")

            if not raw_news:
                self.logger.warning("No news items fetched")
                return []

            # Analyze sentiment
            analyzed_news = self.analyze_sentiment_batch(raw_news)
            self.logger.info(f"Analyzed sentiment for {len(analyzed_news)} news items")

            # Update buffer
            self._update_news_buffer(analyzed_news)

            # Save state
            await self._save_state()

            # Clear sentiment cache
            self._sentiment_cache.clear()

            self.logger.info(
                f"News analysis update complete. Buffer size: {len(self.news_buffer)}"
            )
            return analyzed_news

        except Exception as e:
            self.logger.error(f"Error in news analysis update: {e}")
            return []

    def _update_news_buffer(self, new_items: List[NewsItem]):
        """Update the news buffer with new items."""
        # Add new items to buffer
        self.news_buffer.extend(new_items)

        # Remove duplicates based on URL
        seen_urls = set()
        unique_items = []
        for item in self.news_buffer:
            if item.url not in seen_urls:
                unique_items.append(item)
                seen_urls.add(item.url)

        # Sort by timestamp (newest first) and limit size
        unique_items.sort(key=lambda x: x.timestamp, reverse=True)
        self.news_buffer = unique_items[: self.max_buffer_size]

        self.logger.debug(f"Updated news buffer: {len(self.news_buffer)} items")

    async def get_symbol_sentiment(self, symbol: str) -> float:
        """Get weighted sentiment score for a specific symbol."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{int(datetime.now().timestamp() / self._cache_ttl)}"
            if cache_key in self._sentiment_cache:
                return self._sentiment_cache[cache_key]

            # Normalize symbol
            symbol_key = symbol.replace("/", "").upper()
            underlying = (
                symbol_key.replace("USDT", "").replace("USD", "").replace("EUR", "")
            )

            # Calculate weighted sentiment
            total_sentiment = 0.0
            total_weight = 0.0
            current_time = datetime.now().timestamp()
            matched_items = 0

            for item in self.news_buffer:
                if not item.sentiment or not item.symbols:
                    continue

                # Check if symbol matches
                item_symbols = [s.upper() for s in item.symbols]
                if symbol_key in item_symbols or underlying in item_symbols:
                    matched_items += 1

                    # Calculate time decay
                    hours_old = (current_time - item.timestamp) / 3600
                    time_decay = 0.5 ** (hours_old / 24)  # Half-life of 24 hours

                    # Calculate weight
                    weight = (
                        (item.impact_score or 1.0)
                        * time_decay
                        * (item.confidence or 1.0)
                    )

                    # Apply confidence threshold
                    if (item.confidence or 0) >= self.min_confidence_threshold:
                        total_sentiment += item.sentiment * weight
                        total_weight += weight

            # Calculate final score
            if total_weight > 0:
                sentiment_score = total_sentiment / total_weight
            else:
                sentiment_score = 0.0

            # Cache result
            self._sentiment_cache[cache_key] = sentiment_score

            self.logger.debug(
                f"Symbol sentiment for {symbol}: {sentiment_score:.4f} "
                f"(matched {matched_items} items, total_weight={total_weight:.4f})"
            )

            return sentiment_score

        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {e}")
            return 0.0

    def get_sentiment_summary(self) -> Dict:
        """
        Retourne une synthèse globale du sentiment issu des news :
        - Sentiment global (moyenne pondérée)
        - Nombre total de news analysées
        - Sentiment par symbole le plus fréquent
        - Top news les plus “impactantes” (positives ou négatives)
        """
        try:
            if not self.news_buffer:
                return {
                    "sentiment_global": 0.0,
                    "n_news": 0,
                    "top_symbols": [],
                    "top_news": [],
                }

            total = 0.0
            total_weight = 0.0
            symbol_stats = {}
            for item in self.news_buffer:
                if item.sentiment is None or item.impact_score is None:
                    continue
                weight = item.impact_score
                total += item.sentiment * weight
                total_weight += weight
                for sym in item.symbols:
                    sym = sym.upper()
                    if sym not in symbol_stats:
                        symbol_stats[sym] = {"total": 0.0, "weight": 0.0, "count": 0}
                    symbol_stats[sym]["total"] += item.sentiment * weight
                    symbol_stats[sym]["weight"] += weight
                    symbol_stats[sym]["count"] += 1

            sentiment_global = total / total_weight if total_weight > 0 else 0.0

            top_symbols = []
            if symbol_stats:
                sorted_syms = sorted(
                    symbol_stats.items(), key=lambda x: x[1]["count"], reverse=True
                )[:5]
                for sym, stats in sorted_syms:
                    avg = (
                        stats["total"] / stats["weight"] if stats["weight"] > 0 else 0.0
                    )
                    top_symbols.append(
                        {
                            "symbol": sym,
                            "sentiment": avg,
                            "n_news": stats["count"],
                        }
                    )

            top_news = sorted(
                [
                    item
                    for item in self.news_buffer
                    if item.sentiment is not None and item.impact_score is not None
                ],
                key=lambda x: abs(x.sentiment) * x.impact_score,
                reverse=True,
            )[:3]
            top_news_list = [
                {
                    "title": n.title,
                    "sentiment": n.sentiment,
                    "impact_score": n.impact_score,
                    "symbols": n.symbols,
                    "source": n.source,
                    "url": n.url,
                    "timestamp": n.timestamp,
                }
                for n in top_news
            ]

            return {
                "sentiment_global": sentiment_global,
                "n_news": len(self.news_buffer),
                "top_symbols": top_symbols,
                "top_news": top_news_list,
            }
        except Exception as e:
            self.logger.error(f"Error in get_sentiment_summary: {e}")
            return {
                "sentiment_global": 0.0,
                "n_news": 0,
                "top_symbols": [],
                "top_news": [],
            }
