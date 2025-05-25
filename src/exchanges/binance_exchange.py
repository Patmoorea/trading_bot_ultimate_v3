"""
Binance Exchange Module
Handles all interactions with Binance API (Spot Trading)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timezone
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialise BinanceExchange
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: If True, use testnet instead of mainnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._exchange = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the exchange connection"""
        try:
            self._exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Changed from 'future' to 'spot'
                    'adjustForTimeDifference': True,
                    'testnet': self.testnet
                }
            })
            
            # Configure testnet URLs if needed
            if self.testnet:
                self._exchange.urls['api'] = {
                    'web': 'https://testnet.binance.vision',  # Changed to spot testnet URL
                    'rest': 'https://testnet.binance.vision'
                }
            
            await self._exchange.load_markets()
            self._initialized = True
            logger.info("BinanceExchange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BinanceExchange: {e}")
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
                'last': str(ticker.get('last', 0)),
                'bid': str(ticker.get('bid', 0)),
                'ask': str(ticker.get('ask', 0)),
                'high': str(ticker.get('high', 0)),
                'low': str(ticker.get('low', 0)),
                'baseVolume': str(ticker.get('baseVolume', 0)),
                'quoteVolume': str(ticker.get('quoteVolume', 0)),
                'timestamp': ticker.get('timestamp', 0)
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
                if isinstance(data, dict) and float(data.get('total', 0)) > 0
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    # Les autres méthodes restent similaires, mais nous retirons les méthodes spécifiques aux futures
    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._initialized = False

    async def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> List[List[float]]:
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
            return {
                'bids': orderbook['bids'],
                'asks': orderbook['asks']
            }
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

    async def create_order(self, symbol: str, order_type: str, side: str, 
                         amount: str, price: Optional[str] = None) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Exchange not initialized")
        
        try:
            params = {}
            if float(amount) <= 0:
                raise ValueError("Amount must be positive")
            
            if order_type == 'limit':
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
