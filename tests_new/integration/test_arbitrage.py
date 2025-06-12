import pytest
from decimal import Decimal
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner

def test_arbitrage_scanner_init():
    scanner = ArbitrageScanner([])
    assert scanner is not None
    assert hasattr(scanner, 'exchanges')
    assert isinstance(scanner.min_profit_threshold, Decimal)
