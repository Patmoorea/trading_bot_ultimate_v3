import os
import json
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Optional, Set, Any
import logging
import aiohttp
import ssl
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class SymbolExtractor:
    def __init__(self, symbol_mapping: Optional[Dict[str, str]] = None):
        self.symbol_mapping = symbol_mapping or {
            "bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH", "cardano": "ADA", "ada": "ADA",
            "solana": "SOL", "sol": "SOL", "ripple": "XRP", "xrp": "XRP", "dogecoin": "DOGE", "doge": "DOGE",
            "polkadot": "DOT", "dot": "DOT", "binance": "BNB", "bnb": "BNB", "matic": "MATIC", "polygon": "MATIC",
            "litecoin": "LTC", "ltc": "LTC", "shiba": "SHIB", "shib": "SHIB", "tron": "TRX", "trx": "TRX",
            "avalanche": "AVAX", "avax": "AVAX", "chainlink": "LINK", "link": "LINK", "uniswap": "UNI", "uni": "UNI",
            "stellar": "XLM", "xlm": "XLM",
        }
        self.known_tickers = set(self.symbol_mapping.values())
        self.regex_patterns = [
            (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), ticker)
            for name, ticker in self.symbol_mapping.items()
        ]

    def extract_symbols(self, text: str) -> List[str]:
        found: Set[str] = set()
        if not text:
            return []
        for pattern, ticker in self.regex_patterns:
            if pattern.search(text):
                found.add(ticker)
        # Paires crypto et tickers
        for pair in re.findall(r"\b([A-Z]{3,5})[/-]?(USDT|USD|BTC|ETH)?\b", text.upper()):
            ticker = pair[0]
            if ticker in self.known_tickers:
                found.add(ticker)
        for match in re.findall(r"\$([A-Z]{3,5})\b", text.upper()):
            if match in self.known_tickers:
                found.add(match)
        return list(found)

class NewsSentimentAnalyzer:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.symbol_extractor = SymbolExtractor(config.get("symbol_mapping"))
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self._model = None
        self._tokenizer = None
        self.sources = [
            {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "type": "rss", "weight": 0.9},
            {"name": "CryptoCompare", "url": "https://min-api.cryptocompare.com/data/v2/news/?lang=EN", "type": "json", "weight": 0.7},
            {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss", "type": "rss", "weight": 0.8},
        ]
        self.news_buffer: List[Dict] = []
        self.sentiment_weight = config.get("news", {}).get("sentiment_weight", 0.15)
        self.update_interval = config.get("news", {}).get("update_interval", 300)

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self._model.to(self.device)
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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context), headers=headers) as session:
            tasks = [self._fetch_source(session, source) for source in self.sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_news = []
            for result in results:
                if isinstance(result, list):
                    valid_news.extend(result)
            print(f"\n[NEWS DEBUG] {len(valid_news)} news récupérées ce cycle")
            if valid_news:
                for idx, n in enumerate(valid_news[:5]):
                    print(f"  - {n['source']}: {n['title'][:100]} | Symbols: {n['symbols']}")
            else:
                print("  (Aucune news récupérée)")
            return valid_news

    async def _fetch_source(self, session: aiohttp.ClientSession, source: Dict) -> List[Dict]:
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

    def analyze_sentiment_batch(self, news_items: List[Dict], low_watermark_ratio: float = 0.2) -> List[Dict]:
        """
        Analyse un batch de news avec FinBERT et retourne la liste des news enrichies avec sentiment et impact_score.
        Watermark ratio est forcé à une valeur valide (0.05 à 0.5) pour éviter toute exception.
        Le sentiment est calculé comme score bullish - bearish (positive - negative).
        """
        # Patch ultime : force le watermark ratio à une valeur valide
        try:
            low_watermark_ratio = float(low_watermark_ratio)
        except Exception:
            low_watermark_ratio = 0.2
        if low_watermark_ratio > 0.5 or low_watermark_ratio < 0.05:
            print(f"[DEBUG] Watermark ratio {low_watermark_ratio} is invalid, forcing to 0.2")
            low_watermark_ratio = 0.2
        import inspect
        stack = inspect.stack()
        if len(stack) > 1:
            caller = stack[1]
            print(f"[DEBUG] Called from {caller.filename}:{caller.lineno} with low_watermark_ratio={low_watermark_ratio}")
        else:
            print("[DEBUG] Called from REPL or top-level")

        print("[DEBUG] analyze_sentiment_batch entrée")
        if not news_items:
            print("[SENTIMENT] Aucun article à analyser.")
            return []

        try:
            print("[DEBUG] Début try analyze_sentiment_batch")
            print("[DEBUG] low_watermark_ratio:", low_watermark_ratio)
            print("[DEBUG] news_items:", news_items[:2])
            texts = [f"{item['title']}. {item['text']}"[:512] for item in news_items]
            print("[DEBUG] texts:", texts[:2])
            print("[DEBUG] Nombre de textes à analyser:", len(texts))

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            print("[DEBUG] Inputs batch prêt (device:", self.device, ")")
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            print("[DEBUG] Softmax scores:", scores.tolist()[:5])

            results = []
            for i, item in enumerate(news_items):
                # FinBERT classes: [negative, neutral, positive]
                sentiment = float(scores[i][2] - scores[i][0])  # bullish - bearish
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
            std_sent = np.std([res["sentiment"] for res in results]) if results else 0
            print(
                f"Analyzed {len(results)}/{len(news_items)} items | Mean: {mean_sent:.4f} ± {std_sent:.4f}"
            )
            for i, res in enumerate(results[:5]):
                print(
                    f"  - Sentiment {i+1}: {res['sentiment']:.4f} | Titre: {res['title'][:60]}"
                )
            print("[DEBUG] analyze_sentiment_batch RETURN TYPE:", type(results))
            return results
        except Exception as e:
            print("[DEBUG] EXCEPTION analyze_sentiment_batch:", e)
            print("[DEBUG] Exception details:", repr(e))
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            print("[DEBUG] analyze_sentiment_batch RETURN TYPE:", type([]))
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
                f"News analysis: {len(analyzed_news)} items | "
                f"Mean sentiment: {mean_sentiment:.4f} ± {std_sentiment:.4f}"
            )
            self.news_buffer = [
                *self.news_buffer[-100:],
                *analyzed_news,
            ][-200:]
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

    async def get_symbol_sentiment(self, symbol: str) -> float:
        try:
            print(f"CALLING get_symbol_sentiment: symbol='{symbol}'")
            symbol_key = symbol.replace("/", "").upper()
            underlying = symbol_key.replace("USDT", "").replace("USD", "")
            print(f"[DEBUG INIT] symbol_key='{symbol_key}' underlying='{underlying}'")
            symbol_variants = {
                "BTCUSDT": ["BTC", "BITCOIN", "bitcoin"],
                "ETHUSDT": ["ETH", "ETHEREUM", "ethereum", "ether"],
                "ADAUSDT": ["ADA", "CARDANO", "cardano"],
                "SOLUSDT": ["SOL", "SOLANA", "solana"],
                "BNBUSDT": ["BNB", "BINANCE", "binance"],
                "XRPUSDT": ["XRP", "RIPPLE", "ripple"],
                "DOGEUSDT": ["DOGE", "DOGECOIN", "dogecoin"],
                "AVAXUSDT": ["AVAX", "AVALANCHE", "avalanche"],
                "DOTUSDT": ["DOT", "POLKADOT", "polkadot"],
                "MATICUSDT": ["MATIC", "POLYGON", "polygon"]
            }
            search_terms = symbol_variants.get(symbol_key, [underlying])
            search_terms.extend([symbol_key, underlying])
            search_terms = list(set(search_terms))
            print(f"[DEBUG SEARCH] Termes de recherche pour {symbol_key}: {search_terms}")
            total = 0.0
            total_weight = 0.0
            current_time = datetime.now().timestamp()
            matched = False
            print(
                f"[DEBUG SENTIMENT] Recherche des news pour {search_terms}... Buffer size: {len(self.news_buffer)}"
            )
            for news in self.news_buffer:
                news_symbols = news.get("symbols", [])
                title = news.get("title", "").lower()
                text = news.get("text", "").lower()
                content = f"{title} {text}"
                if news_symbols:
                    print(
                        f"[DEBUG NEWS] Title: {news.get('title', '')[:60]} | Symbols: {news_symbols}"
                    )
                match_extracted = any(
                    s.upper().strip() in [term.upper() for term in search_terms]
                    for s in news_symbols
                )
                match_content = any(
                    term.lower() in content for term in search_terms
                )
                if match_extracted or match_content:
                    matched = True
                    hours_old = (
                        current_time - news.get("timestamp", current_time)
                    ) / 3600
                    decay = 0.5 ** (hours_old / 24)
                    sentiment = news.get("sentiment", 0)
                    impact = news.get("impact_score", 1) or 1
                    match_type = "extracted" if match_extracted else "content"
                    print(
                        f"[DEBUG MATCH] ({match_type}) Titre: {news.get('title', '')[:60]} | Symbols: {news_symbols} | sentiment={sentiment:.4f} | impact={impact:.2f} | decay={decay:.2f}"
                    )
                    total += sentiment * impact * decay
                    total_weight += impact * decay
            if not matched:
                print(
                    f"[DEBUG SENTIMENT] Aucun match trouvé pour {search_terms}"
                )
                available_symbols = set()
                for news in self.news_buffer[:5]:
                    available_symbols.update(news.get("symbols", []))
                print(
                    f"[DEBUG SENTIMENT] Symboles disponibles (échantillon): {list(available_symbols)[:10]}"
                )
            score = total / max(total_weight, 1e-6) if total_weight > 0 else 0.0
            print(
                f"[DEBUG SENTIMENT] symbol={symbol_key} | sentiment_score={score:.4f} | total={total:.4f} | total_weight={total_weight:.4f} | matched={matched}"
            )
            return score
        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return 0.0

    async def _save_state(self, data: Optional[Dict] = None):
        path = self.config.get("news", {}).get(
            "storage_path", "data/news_analysis.json"
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        **({"analysis": data} if data else {}),
                        "buffer_size": len(self.news_buffer),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self.logger.error(f"Save failed: {str(e)}")

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
        impact = news.get("source_weight", 1.0)
        impact *= 1.0 + min(1.0, abs(sentiment))
        hours_old = (
            datetime.now().timestamp()
            - news.get("timestamp", datetime.now().timestamp())
        ) / 3600
        impact *= max(0.1, 1.0 - (hours_old / 48))
        n_symbols = len(news.get("symbols", []))
        impact *= min(2.0, 1.0 + n_symbols * 0.1)
        return max(0.1, min(5.0, impact))

    def get_sentiment_summary(self) -> Dict[str, Any]:
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
                sentiment = item.get("sentiment")
                impact_score = item.get("impact_score", 1.0)
                if sentiment is None or impact_score is None:
                    continue
                weight = impact_score
                total += sentiment * weight
                total_weight += weight
                for sym in item.get("symbols", []):
                    sym = sym.upper()
                    if sym not in symbol_stats:
                        symbol_stats[sym] = {"total": 0.0, "weight": 0.0, "count": 0}
                    symbol_stats[sym]["total"] += sentiment * weight
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
            valid_news = [
                item
                for item in self.news_buffer
                if item.get("sentiment") is not None
                and item.get("impact_score") is not None
            ]
            top_news = sorted(
                valid_news,
                key=lambda x: abs(x.get("sentiment", 0)) * x.get("impact_score", 1),
                reverse=True,
            )[:3]
            top_news_list = [
                {
                    "title": n.get("title", ""),
                    "sentiment": n.get("sentiment", 0),
                    "impact_score": n.get("impact_score", 1),
                    "symbols": n.get("symbols", []),
                    "source": n.get("source", ""),
                    "url": n.get("url", ""),
                    "timestamp": n.get("timestamp", 0),
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

def extract_symbols(title: str) -> List[str]:
    """Legacy function for backward compatibility."""
    extractor = SymbolExtractor()
    return extractor.extract_symbols(title)