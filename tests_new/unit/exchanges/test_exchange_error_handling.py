import pytest
from src.core.exchange import ExchangeInterface
from tests_new.base_test import BaseTest
import ccxt

class TestExchangeErrorHandling(BaseTest):
    @pytest.fixture
    def exchange(self):
        ex = ExchangeInterface('binance', testnet=True)
        ex.set_test_mode(True)
        return ex

    def test_invalid_exchange(self):
        """Test initialization with invalid exchange"""
        with pytest.raises(AttributeError):
            ExchangeInterface('invalid_exchange')

    def test_invalid_symbol(self):
        """Test operations with invalid symbol"""
        exchange = ExchangeInterface('binance', testnet=True)
        with pytest.raises(Exception):
            exchange.get_ticker('INVALID/PAIR')

    def test_connection_failure(self, exchange):
        """Test connection failure handling"""
        assert not exchange.check_connection()

    def test_market_load_failure(self, exchange):
        """Test market loading failure"""
        with pytest.raises(Exception):
            exchange.load_markets()

    def test_amount_format_failure(self, exchange):
        """Test amount formatting failure"""
        invalid_values = [
            ('nan', "Amount must be a finite number"),
            (float('inf'), "Amount must be a finite number"),
            (-1.0, "Amount must be positive"),
            ("not a number", "Amount must be a number"),
            ([1, 2, 3], "Amount must be a number"),
            ({"amount": 1}, "Amount must be a number"),
            (None, "Amount must be a number"),
        ]
        
        for value, expected_msg in invalid_values:
            with pytest.raises(ValueError) as exc_info:
                exchange.format_amount('BTC/USDT', value)
            assert str(exc_info.value).startswith("Invalid amount format for BTC/USDT")

    def test_min_order_amount_invalid_symbol(self, exchange):
        """Test minimum order amount with invalid symbol"""
        with pytest.raises(Exception):
            exchange.get_min_order_amount('INVALID/PAIR')

    @pytest.mark.skip(reason="Requires actual API credentials")
    def test_invalid_order(self, exchange):
        """Test invalid order creation"""
        with pytest.raises(Exception):
            exchange.create_order(
                symbol='BTC/USDT',
                order_type='limit',
                side='buy',
                amount=0.0000001,  # Too small amount
                price=30000.0
            )
