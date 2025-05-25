"""
Integration tests for Gate.io Exchange Module
Created by: Patmoorea
Last updated: 2024-01-19 23:23:15 UTC
"""

import pytest
import asyncio
import time
import os
from decimal import Decimal
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from src.exchanges.gateio_exchange import GateIOExchange
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestConstants:
    """Test constants and configuration"""
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "1h"
    CONTRACT_SIZE = 0.001
    DEFAULT_TIMEOUT = 5.0
    LAST_UPDATE = "2024-01-19 23:23:15"
    AUTHOR = "Patmoorea"
    
    TEST_API_KEY = os.getenv('GATEIO_API_KEY')
    TEST_API_SECRET = os.getenv('GATEIO_API_SECRET')

def run_async(coro):
    """Helper to run coroutine in synchronous context."""
    return asyncio.get_event_loop().run_until_complete(coro)

class TestGateIOExchange:
    """Test suite for GateIOExchange"""

    def setup_method(self, method):
        """Set up test cases."""
        logger.info(f"Starting test setup at {TestConstants.LAST_UPDATE} by {TestConstants.AUTHOR}")
        
        if not all([TestConstants.TEST_API_KEY, TestConstants.TEST_API_SECRET]):
            pytest.skip("Gate.io credentials not configured in environment variables")
            
        self.exchange = GateIOExchange(
            api_key=TestConstants.TEST_API_KEY,
            api_secret=TestConstants.TEST_API_SECRET
        )
        run_async(self.exchange.initialize())

    def teardown_method(self, method):
        """Clean up after test cases."""
        if hasattr(self, 'exchange'):
            run_async(self.exchange.close())
        logger.info(f"Test cleanup completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    @pytest.mark.integration
    def test_market_data(self):
        """Test la récupération des données de marché"""
        async def run_test():
            logger.info(f"Running market data test by {TestConstants.AUTHOR}")
            
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
            logger.info(f"Running account data test by {TestConstants.AUTHOR}")
            
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
            if 'USDT' in balance:
                logger.info(f"USDT Balance: {balance['USDT']}")

        run_async(run_test())

    @pytest.mark.integration
    def test_order_lifecycle(self):
        """Test le cycle de vie complet d'un ordre spot"""
        async def run_test():
            logger.info(f"Running order lifecycle test by {TestConstants.AUTHOR}")
            
            symbol = TestConstants.SYMBOL
            
            # 1. Vérifier le solde disponible
            balance = await self.exchange.get_balance()
            usdt_balance = float(balance.get('USDT', {}).get('free', 0))
            logger.info(f"Available USDT balance: {usdt_balance}")

            if usdt_balance < 5:
                logger.warning(f"Insufficient USDT balance for testing. Need at least 5 USDT")
                pytest.skip("Insufficient funds for order testing")
                return

            # 2. Vérifier le prix du marché
            ticker = await self.exchange.get_ticker(symbol)
            price = float(ticker["last"])
            logger.info(f"Current market price: {price}")
            
            # 3. Calculer la taille de position pour un ordre minuscule
            position_size = round((5 / price) * 0.01, 6)  # Utilise 1% de 5 USDT
            
            try:
                # Créer un ordre limit
                order = await self.exchange.create_order(
                    symbol=symbol,
                    order_type="limit",
                    side="buy",
                    amount=str(position_size),
                    price=str(price * 0.9),  # 10% sous le marché
                )
                assert order["id"] is not None
                order_id = order["id"]
                logger.info(f"Order created: {order}")

                # Vérifier l'ordre
                fetched_order = await self.exchange.get_order(symbol, order_id)
                assert fetched_order["id"] == order_id
                assert fetched_order["status"] in ["open", "created"]
                logger.info(f"Fetched order: {fetched_order}")

                # Annuler l'ordre
                cancelled = await self.exchange.cancel_order(symbol, order_id)
                assert cancelled is True
                logger.info("Order cancelled successfully")

            except Exception as e:
                logger.error(f"Error during order lifecycle test: {str(e)}")
                raise

        run_async(run_test())

    @pytest.mark.integration
    def test_arbitrage_opportunities(self):
        """Test la détection d'opportunités d'arbitrage"""
        async def run_test():
            logger.info(f"Running arbitrage opportunities test by {TestConstants.AUTHOR}")
            
            symbols = ["BTC/USDT", "ETH/USDT", "LTC/USDT"]
            opportunities = await self.exchange.get_arbitrage_opportunities(
                symbols=symbols,
                min_profit_percent=0.1
            )
            
            assert isinstance(opportunities, list)
            for opp in opportunities:
                assert 'symbol' in opp
                assert 'ask_price' in opp
                assert 'bid_price' in opp
                assert 'spread' in opp
                assert 'fees' in opp
                assert 'potential_profit' in opp
                assert float(opp['potential_profit']) >= 0

            logger.info(f"Found {len(opportunities)} arbitrage opportunities")

        run_async(run_test())

    @pytest.mark.integration
    def test_trading_fees(self):
        """Test la récupération des frais de trading"""
        async def run_test():
            logger.info(f"Running trading fees test by {TestConstants.AUTHOR}")
            
            symbol = TestConstants.SYMBOL
            fees = self.exchange._get_total_fees(symbol)
            
            assert isinstance(fees, float)
            assert fees > 0
            assert fees < 1.0  # Les frais devraient être inférieurs à 100%
            
            logger.info(f"Trading fees for {symbol}: {fees}%")

        run_async(run_test())

    @pytest.mark.integration
    def test_min_amounts(self):
        """Test la récupération des montants minimums"""
        async def run_test():
            logger.info(f"Running minimum amounts test by {TestConstants.AUTHOR}")
            
            symbol = TestConstants.SYMBOL
            min_amount = self.exchange._min_trade_amounts.get(symbol)
            
            assert min_amount is not None
            assert float(min_amount) > 0
            
            logger.info(f"Minimum trade amount for {symbol}: {min_amount}")

        run_async(run_test())

    @pytest.mark.integration
    def test_error_handling(self):
        """Test la gestion des erreurs"""
        async def run_test():
            logger.info(f"Running error handling test by {TestConstants.AUTHOR}")
            
            # Test avec un symbol invalide
            with pytest.raises(Exception):
                await self.exchange.get_ticker("INVALID/PAIR")
            
            # Test avec un order_id invalide
            with pytest.raises(Exception):
                await self.exchange.get_order(TestConstants.SYMBOL, "invalid_order_id")

            logger.info("Error handling tests passed successfully")

        run_async(run_test())

    @pytest.mark.integration
    def test_market_data_consistency(self):
        """Test la cohérence des données de marché"""
        async def run_test():
            logger.info(f"Running market data consistency test by {TestConstants.AUTHOR}")
            
            symbol = TestConstants.SYMBOL
            
            # Récupérer ticker et orderbook
            ticker = await self.exchange.get_ticker(symbol)
            orderbook = await self.exchange.get_orderbook(symbol)
            
            # Vérifier la cohérence des prix
            assert float(ticker['bid']) <= float(ticker['last']) <= float(ticker['ask'])
            assert len(orderbook['bids']) > 0 and len(orderbook['asks']) > 0
            assert float(orderbook['bids'][0][0]) < float(orderbook['asks'][0][0])
            
            logger.info("Market data consistency verified")

        run_async(run_test())

