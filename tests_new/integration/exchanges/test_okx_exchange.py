"""
Integration tests for OKX Exchange Module (Futures Trading)
Created by: Patmoorea
Last updated: 2025-05-19 20:04:02 UTC
"""

import pytest
import asyncio
import time
import os
from decimal import Decimal
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from src.exchanges.okx_exchange import OKXExchange
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestConstants:
    """
    Test constants and configuration
    Updated by: Patmoorea on 2025-05-19
    """
    SYMBOL = "BTC/USDT:USDT"
    TIMEFRAME = "1h"
    LEVERAGE = 5
    CONTRACT_SIZE = 0.001
    DEFAULT_TIMEOUT = 5.0
    LAST_UPDATE = "2025-05-19 20:04:02"
    AUTHOR = "Patmoorea"
    
    # Utiliser les bons noms de variables
    TEST_API_KEY = os.getenv('OKX_API_KEY')
    TEST_API_SECRET = os.getenv('OKX_API_SECRET')
    TEST_API_PASSPHRASE = os.getenv('OKX_API_PASSPHRASE')

def run_async(coro):
    """Helper to run coroutine in synchronous context."""
    return asyncio.get_event_loop().run_until_complete(coro)

class TestOKXExchange:
    """
    Test suite for OKXExchange
    Author: Patmoorea
    Last Update: 2025-05-19 20:04:02
    """

    def setup_method(self, method):
        """Set up test cases."""
        logger.info(f"Starting test setup at {TestConstants.LAST_UPDATE} by {TestConstants.AUTHOR}")
        
        if not all([TestConstants.TEST_API_KEY, 
                   TestConstants.TEST_API_SECRET, 
                   TestConstants.TEST_API_PASSPHRASE]):
            pytest.skip("OKX credentials not configured in environment variables")
            
        self.exchange = OKXExchange(
            api_key=TestConstants.TEST_API_KEY,
            api_secret=TestConstants.TEST_API_SECRET,
            password=TestConstants.TEST_API_PASSPHRASE
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

            # Test des positions ouvertes
            positions = await self.exchange.get_positions()
            assert isinstance(positions, list)
            logger.info(f"Open positions: {positions}")

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
    def test_futures_settings(self):
        """Test la configuration des paramètres futures"""
        async def run_test():
            logger.info(f"Running futures settings test by {TestConstants.AUTHOR}")
            
            # Activer le mode hedge
            hedge_set = await self.exchange.set_position_mode(True)
            assert hedge_set
            logger.info("Hedge mode enabled")

            # Définir le levier
            leverage_set = await self.exchange.set_leverage(
                symbol=TestConstants.SYMBOL,
                leverage=TestConstants.LEVERAGE,
                params={'marginMode': 'cross'}
            )
            assert leverage_set
            logger.info(f"Leverage set to {TestConstants.LEVERAGE}x with cross margin")

            # Vérifier les positions actuelles
            positions = await self.exchange.get_positions()
            for pos in positions:
                logger.info(f"Current position: {pos}")

        run_async(run_test())

    @pytest.mark.integration
    def test_order_lifecycle(self):
        """Test le cycle de vie complet d'un ordre futures"""
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
            
            # 3. Calculer la taille de position
            contract_value = price * TestConstants.CONTRACT_SIZE
            max_position_size = (usdt_balance * 0.95 * TestConstants.LEVERAGE) / contract_value
            position_size = min(max_position_size, TestConstants.CONTRACT_SIZE)
            position_size = round(position_size, 3)
            
            try:
                # Paramètres spécifiques à OKX
                params = {
                    'tdMode': 'cross',
                    'posSide': 'long'
                }
                
                order = await self.exchange.create_order(
                    symbol=symbol,
                    order_type="limit",
                    side="buy",
                    amount=str(position_size),
                    price=str(price * 0.9),  # 10% sous le marché
                    params=params
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
                raise

        run_async(run_test())

    @pytest.mark.integration
    def test_available_futures(self):
        """Test pour afficher les contrats futures disponibles et leurs limites"""
        async def run_test():
            logger.info(f"Running available futures test by {TestConstants.AUTHOR}")
            
            markets = self.exchange._exchange.markets
            logger.info("Available futures contracts and their limits:")
            for symbol, market in markets.items():
                if ':USDT' in symbol:  # Ne montrer que les contrats futures USDT
                    logger.info(f"\nSymbol: {symbol}")
                    logger.info(f"Contract type: {market.get('type')}")
                    logger.info(f"Limits: {market.get('limits', {})}")
                    logger.info(f"Precision: {market.get('precision', {})}")
                    logger.info(f"Leverage range: {market.get('leverage', {})}")

        run_async(run_test())
