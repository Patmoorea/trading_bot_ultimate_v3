"""
Arbitrage Scanner Core Module
Version 1.0.0 - Patmoorea - 2025-05-24
"""

from decimal import Decimal
from datetime import datetime
from typing import List, Dict, Any

class ArbitrageScanner:
    def __init__(
        self,
        exchanges=None,
        min_profit_threshold: Decimal = Decimal("0.002"),
        max_price_deviation: Decimal = Decimal("0.05"),
    ):
        self.exchanges = exchanges or []
        self.min_profit_threshold = min_profit_threshold
        self.max_price_deviation = max_price_deviation
        self.quote_currency_map = {
            "binance": "USDC",
            "bybit": "USDT",
            "okx": "USDT",
            "gateio": "USDT",
        }

    def scan(self) -> List[Dict[str, Any]]:
        """
        Mock scan method for now — returns one dummy opportunity
        """
        return [{
            "symbol": "BTC/USDT",
            "buy_exchange": "binance",
            "sell_exchange": "bybit",
            "buy_price": 30000,
            "sell_price": 30300,
            "profit_pct": 0.01,
            "timestamp": datetime.utcnow().isoformat()
        }]

    def _validate_symbol(self, symbol: str, exchange: str) -> bool:
        """Valide si un symbole est compatible avec l'exchange"""
        quote = self._get_exchange_quote_currency(exchange)
        return quote in symbol and '/' in symbol

    def _get_equivalent_symbol(self, symbol: str, exchange: str) -> str:
        """Convertit le symbole vers le format de l'exchange"""
        base, quote = symbol.split('/')
        expected_quote = self._get_exchange_quote_currency(exchange)
        return f"{base}/{expected_quote}"

    def _calculate_profit(self, buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """Calcule la rentabilité"""
        return (sell_price - buy_price) / buy_price

    def _check_price_deviation(self, prices: list[Decimal]) -> bool:
        """Vérifie si l'écart de prix est acceptable"""
        min_price = min(prices)
        max_price = max(prices)
        deviation = (max_price - min_price) / min_price
        return deviation <= self.max_price_deviation

    def _get_exchange_quote_currency(self, exchange: str) -> str:
        """Retourne la devise de cotation de l'exchange"""
        return self.quote_currency_map.get(exchange.lower(), "USDT")

    async def scan_opportunities(self, symbols: list[str]) -> list[dict]:
        """Scanne les opportunités d'arbitrage"""
        opportunities = []
        for symbol in symbols:
            valid_exchanges = [
                e for e in self.exchanges if self._validate_symbol(symbol, e.name)
            ]
            for buy in valid_exchanges:
                for sell in valid_exchanges:
                    if buy.name == sell.name:
                        continue
                    buy_sym = self._get_equivalent_symbol(symbol, buy.name)
                    sell_sym = self._get_equivalent_symbol(symbol, sell.name)
                    buy_ticker = await buy.get_ticker(buy_sym)
                    sell_ticker = await sell.get_ticker(sell_sym)

                    buy_price = buy_ticker['ask']
                    sell_price = sell_ticker['bid']
                    profit = self._calculate_profit(buy_price, sell_price)

                    if profit >= self.min_profit_threshold:
                        opportunities.append({
                            "symbol": symbol,
                            "buy_exchange": buy.name,
                            "sell_exchange": sell.name,
                            "buy_price": float(buy_price),
                            "sell_price": float(sell_price),
                            "profit_pct": float(profit),
                            "timestamp": sell_ticker['timestamp']
                        })
        return opportunities
