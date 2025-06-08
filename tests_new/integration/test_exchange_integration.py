import pytest
from src.core.exchange import ExchangeInterface
from tests_new.base_test import BaseTest
import ccxt

class TestExchangeIntegration(BaseTest):
    @pytest.fixture
    def exchange(self):
        """Create test exchange instance"""
        ex = ExchangeInterface(
            exchange_id='binance',
            testnet=True
        )
        ex.set_test_mode(True)
        return ex

    def test_exchange_initialization(self, exchange):
        """Test exchange initialization"""
        assert exchange.exchange_id == 'binance'
        assert exchange.testnet is True
        assert isinstance(exchange.exchange, ccxt.Exchange)

    def test_market_loading(self, exchange):
        """Test market data loading"""
        markets = exchange.markets
        assert isinstance(markets, dict)
        assert len(markets) > 0
        assert 'BTC/USDT' in markets

    def test_amount_formatting(self, exchange):
        """Test amount formatting"""
        # Test avec une valeur valide
        formatted = exchange.format_amount('BTC/USDT', 0.12345678)
        assert isinstance(formatted, float)
        assert formatted == 0.12345678  # Should match precision of 8

        # Test avec une valeur qui nécessite un arrondi
        formatted = exchange.format_amount('BTC/USDT', 0.123456789)
        assert formatted == 0.12345679  # Arrondi à 8 décimales

    def test_min_order_amount(self, exchange):
        """Test minimum order amount fetching"""
        min_amount = exchange.get_min_order_amount('BTC/USDT')
        assert isinstance(min_amount, float)
        assert min_amount == 0.0001  # Value from test market data

    @pytest.mark.skip(reason="Requires actual API credentials")
    def test_balance_fetching(self, exchange):
        """Test balance fetching"""
        balance = exchange.get_balance()
        assert isinstance(balance, dict)
        assert 'total' in balance
        assert 'free' in balance
        assert 'used' in balance
