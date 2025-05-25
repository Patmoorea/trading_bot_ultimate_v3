"""
Integration tests for Binance Exchange Module (Spot Trading)
"""

import pytest
import asyncio
import time
import os
from decimal import Decimal
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from src.exchanges.binance_exchange import BinanceExchange
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestConstants:
    """Test constants and configuration"""
    SYMBOL = "BTC/USDC"
    TIMEFRAME = "1h"
    MIN_ORDER_VALUE = 10  # Valeur minimum en USDC
    DEFAULT_TIMEOUT = 5.0
    
    # Utilisation des clés depuis .env
    TEST_API_KEY = os.getenv('BINANCE_API_KEY')
    TEST_API_SECRET = os.getenv('BINANCE_API_SECRET')

def run_async(coro):
    """Helper to run coroutine in synchronous context."""
    return asyncio.get_event_loop().run_until_complete(coro)

class TestBinanceExchange:
    """Test suite for BinanceExchange"""

    def setup_method(self, method):
        """Set up test cases."""
        self.exchange = BinanceExchange(
            api_key=TestConstants.TEST_API_KEY,
            api_secret=TestConstants.TEST_API_SECRET,
            testnet=False
        )
        run_async(self.exchange.initialize())

    def teardown_method(self, method):
        """Clean up after test cases."""
        if hasattr(self, 'exchange'):
            run_async(self.exchange.close())

    @pytest.mark.integration
    def test_market_data(self):
        """Test la récupération des données de marché"""
        async def run_test():
            # Test du ticker
            ticker = await self.exchange.get_ticker(TestConstants.SYMBOL)
            assert ticker is not None
            assert "last" in ticker
            assert "bid" in ticker
            assert "ask" in ticker
            assert "baseVolume" in ticker
            assert float(ticker["last"]) >= 0
            
            logger.info(f"Ticker data: {ticker}")

            # Test des klines
            klines = await self.exchange.get_klines(
                symbol=TestConstants.SYMBOL,
                timeframe=TestConstants.TIMEFRAME,
                limit=100
            )
            assert len(klines) > 0
            assert all(len(k) >= 5 for k in klines)
            
            logger.info(f"First kline: {klines[0]}")

            # Test du carnet d'ordres
            orderbook = await self.exchange.get_orderbook(TestConstants.SYMBOL)
            assert "bids" in orderbook
            assert "asks" in orderbook
            assert len(orderbook["bids"]) > 0
            assert len(orderbook["asks"]) > 0
            
            logger.info(f"Top bid: {orderbook['bids'][0]}, Top ask: {orderbook['asks'][0]}")

        run_async(run_test())

    @pytest.mark.integration
    def test_account_data(self):
        """Test la récupération des données du compte"""
        async def run_test():
            # Test du solde
            balance = await self.exchange.get_balance()
            assert isinstance(balance, dict)
            logger.info(f"Account balance: {balance}")

            # Test de l'historique des trades
            trades = await self.exchange.get_my_trades(TestConstants.SYMBOL)
            assert isinstance(trades, list)
            if trades:
                logger.info(f"Latest trade: {trades[-1]}")

            # Afficher les soldes spécifiques
            if 'USDC' in balance:
                logger.info(f"USDC Balance: {balance['USDC']}")
            if 'BTC' in balance:
                logger.info(f"BTC Balance: {balance['BTC']}")

        run_async(run_test())

    @pytest.mark.integration
    def test_order_lifecycle(self):
        """Test le cycle de vie complet d'un ordre"""
        async def run_test():
            symbol = TestConstants.SYMBOL
            
            # 1. Vérifier le solde disponible
            balance = await self.exchange.get_balance()
            usdc_balance = float(balance.get('USDC', {}).get('free', 0))
            logger.info(f"Available USDC balance: {usdc_balance}")

            if usdc_balance < TestConstants.MIN_ORDER_VALUE:
                logger.warning(f"Insufficient USDC balance for testing. Need at least {TestConstants.MIN_ORDER_VALUE} USDC")
                pytest.skip("Insufficient funds for order testing")
                return

            # 2. Vérifier le prix du marché
            ticker = await self.exchange.get_ticker(symbol)
            price = float(ticker["last"])
            logger.info(f"Current market price: {price}")
            
            # 3. Calculer le montant maximum que nous pouvons trader
            max_possible_amount = (usdc_balance * 0.95) / price  # Utilise 95% du solde disponible
            amount = min(max_possible_amount, 0.001)  # Utilise le minimum entre le maximum possible et 0.001 BTC
            amount = round(amount, 6)  # Arrondir à 6 décimales pour BTC
            
            # 4. Placer un ordre limit très bas pour éviter l'exécution
            limit_price = price * 0.5  # 50% sous le marché
            logger.info(f"Placing limit order: amount={amount}, price={limit_price}")
            
            try:
                order = await self.exchange.create_order(
                    symbol=symbol,
                    order_type="limit",
                    side="buy",
                    amount=str(amount),
                    price=str(limit_price)
                )
                assert order["id"] is not None
                order_id = order["id"]
                logger.info(f"Order created: {order}")

                # 5. Vérifier l'ordre
                fetched_order = await self.exchange.get_order(symbol, order_id)
                assert fetched_order["id"] == order_id
                assert fetched_order["status"] in ["open", "created"]
                logger.info(f"Fetched order: {fetched_order}")

                # 6. Annuler l'ordre
                cancelled = await self.exchange.cancel_order(symbol, order_id)
                assert cancelled is True
                logger.info("Order cancelled successfully")

            except Exception as e:
                logger.error(f"Error during order lifecycle test: {str(e)}")
                markets = self.exchange._exchange.markets
                if symbol in markets:
                    logger.info(f"Market limits for {symbol}: {markets[symbol]['limits']}")
                raise

        run_async(run_test())

    @pytest.mark.integration
    def test_available_markets(self):
        """Test pour afficher les marchés disponibles et leurs limites"""
        async def run_test():
            markets = self.exchange._exchange.markets
            logger.info("Available markets and their limits:")
            for symbol, market in markets.items():
                if 'USDC' in symbol:  # Ne montrer que les paires USDC
                    logger.info(f"\nSymbol: {symbol}")
                    logger.info(f"Limits: {market.get('limits', {})}")
                    logger.info(f"Precision: {market.get('precision', {})}")

        run_async(run_test())

