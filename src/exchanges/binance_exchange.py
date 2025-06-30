"""
Binance Exchange Module
Handles all interactions with Binance API (Spot Trading)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import ccxt.async_support as ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._exchange = None
        self._initialized = False

    async def initialize(self):
        try:
            # Création de l'exchange ccxt
            self._exchange = ccxt.binance(
                {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "spot",
                        "adjustForTimeDifference": True,
                        "testnet": self.testnet,
                    },
                }
            )
            logger.info(
                f"BinanceExchange: testnet={self.testnet}, defaultType={self._exchange.options.get('defaultType')}"
            )
            if self.testnet:
                logger.warning("Binance testnet: PAS de load_markets !")
                self._exchange.urls["api"] = {
                    "web": "https://testnet.binance.vision",
                    "rest": "https://testnet.binance.vision",
                }
                # PATCH: override load_markets pour éviter tout appel accidentel
                self._exchange.load_markets = lambda *a, **k: None
            else:
                logger.info(">>> AVANT load_markets")
                await self._exchange.load_markets()
                logger.info(">>> APRES load_markets")

            self._initialized = True
            logger.info("BinanceExchange initialized successfully")

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to initialize BinanceExchange: {e}\n{traceback.format_exc()}"
            )
            raise

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
        Returns:
            Dict containing ticker data
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return {
                "last": str(ticker.get("last", 0)),
                "bid": str(ticker.get("bid", 0)),
                "ask": str(ticker.get("ask", 0)),
                "high": str(ticker.get("high", 0)),
                "low": str(ticker.get("low", 0)),
                "baseVolume": str(ticker.get("baseVolume", 0)),
                "quoteVolume": str(ticker.get("quoteVolume", 0)),
                "timestamp": ticker.get("timestamp", 0),
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    async def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance
        Returns:
            Dict containing balance information
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            balance = await self._exchange.fetch_balance()
            # Filter only assets with non-zero balances
            return {
                currency: data
                for currency, data in balance.items()
                if isinstance(data, dict) and float(data.get("total", 0)) > 0
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._initialized = False

    async def get_klines(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[List[float]]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            klines = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return klines
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            raise

    async def get_orderbook(self, symbol: str) -> Dict[str, List[List[float]]]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            orderbook = await self._exchange.fetch_order_book(symbol)
            return {"bids": orderbook["bids"], "asks": orderbook["asks"]}
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            raise

    async def get_my_trades(self, symbol: str) -> List[Dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            trades = await self._exchange.fetch_my_trades(symbol)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: str,
        price: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            params = {}
            if float(amount) <= 0:
                raise ValueError("Amount must be positive")
            if order_type == "limit":
                if not price or float(price) <= 0:
                    raise ValueError("Valid price required for limit orders")
                order = await self._exchange.create_limit_order(
                    symbol, side, float(amount), float(price), params
                )
            else:
                order = await self._exchange.create_market_order(
                    symbol, side, float(amount), None, params
                )
            return order
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            raise

    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            return order
        except Exception as e:
            logger.error(f"Error fetching order {order_id}: {e}")
            raise

    async def get_historical_data(
        self, pairs: List[str], timeframes: List[str], period: str
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Récupère les données OHLCV historiques pour chaque paire/timeframe sur la période demandée.
        Retourne {timeframe: {pair: pd.DataFrame}}
        PATCH: Ne retourne jamais None, toujours un DataFrame (éventuellement vide), jamais d'objet non-attendu.
        PATCH: Garantit que fetch_ohlcv est toujours awaitable, même en testnet ou mock.
        """
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        result = {}
        try:
            # Parse period (ex: "7d" -> 7 jours)
            if period.endswith("d"):
                days = int(period.replace("d", ""))
                since = int(
                    (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
                )
            else:
                # Fallback: 1 jour
                since = int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000)
            for tf in timeframes:
                tf_result = {}
                for pair in pairs:
                    try:
                        fetch_ohlcv = getattr(self._exchange, "fetch_ohlcv", None)
                        klines = None
                        # --- PATCH: Robust await handling ---
                        if fetch_ohlcv is not None:
                            # Si on a une cofunc, on await ; sinon, on wrap dans une coroutine
                            if asyncio.iscoroutinefunction(fetch_ohlcv):
                                klines = await fetch_ohlcv(pair, tf, since=since)
                            else:
                                # On force le résultat dans une coroutine si jamais il n'est pas async
                                async def fake_coro(*args, **kwargs):
                                    return fetch_ohlcv(*args, **kwargs)
                                klines = await fake_coro(pair, tf, since=since)
                        else:
                            logger.error(f"fetch_ohlcv n'est pas disponible pour {pair} {tf}")
                            klines = []
                        if not klines or len(klines) == 0:
                            logger.error(f"Aucune donnée historique pour {pair} {tf}")
                            tf_result[pair] = pd.DataFrame(
                                columns=[
                                    "timestamp",
                                    "open",
                                    "high",
                                    "low",
                                    "close",
                                    "volume",
                                ]
                            )
                            continue

                        df = pd.DataFrame(
                            klines,
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ],
                        )
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        tf_result[pair] = df
                    except Exception as e:
                        logger.error(f"Erreur historique {pair} {tf}: {e}")
                        tf_result[pair] = pd.DataFrame(
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                        )
                result[tf] = tf_result
            return result
        except Exception as e:
            logger.error(f"Erreur get_historical_data: {e}")
            # PATCH: Ne jamais raise ici, toujours retourner un dict avec DataFrame vides
            for tf in timeframes:
                if tf not in result:
                    result[tf] = {}
                for pair in pairs:
                    if pair not in result[tf]:
                        result[tf][pair] = pd.DataFrame(
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                        )
            return result
