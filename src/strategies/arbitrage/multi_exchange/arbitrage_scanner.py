"""
Arbitrage Scanner Module
Version 1.0.0 - Created: 2025-05-19 03:48:05 by Patmoorea
"""

from typing import List, Dict, Any
import logging
from decimal import Decimal
import asyncio
from datetime import datetime
import numpy as np
from src.utils.datetime_utils import get_utc_now, format_timestamp

class ArbitrageScanner:
    """Scanner for arbitrage opportunities across multiple exchanges"""
    
    def __init__(self, exchanges: List[Any], min_profit_threshold: Decimal = Decimal('0.001'), 
                 max_price_deviation: Decimal = Decimal('0.05')):
        """Initialize arbitrage scanner"""
        self.exchanges = exchanges
        self._min_profit_threshold = min_profit_threshold
        self._max_price_deviation = max_price_deviation
        self._quote_currencies = {
            'binance': 'USDC',  # Binance uses USDC
            'default': 'USDT'   # Other exchanges use USDT
        }
        logging.info(f"ArbitrageScanner initialized with {len(exchanges)} exchanges")

    @property
    def min_profit_threshold(self) -> Decimal:
        """Get minimum profit threshold"""
        return self._min_profit_threshold

    @min_profit_threshold.setter
    def min_profit_threshold(self, value: Decimal):
        """Set minimum profit threshold"""
        value = Decimal(str(value))
        if value < 0:
            raise ValueError("Profit threshold must be non-negative")
        self._min_profit_threshold = value

    @property
    def max_price_deviation(self) -> Decimal:
        """Get maximum price deviation threshold"""
        return self._max_price_deviation

    @max_price_deviation.setter
    def max_price_deviation(self, value: Decimal):
        """Set maximum price deviation threshold"""
        value = Decimal(str(value))
        if value <= 0 or value > 1:
            raise ValueError("Price deviation must be between 0 and 1")
        self._max_price_deviation = value

    def _get_exchange_quote_currency(self, exchange_name: str) -> str:
        """Get the appropriate quote currency for an exchange"""
        exchange_name = exchange_name.lower()
        return self._quote_currencies.get(exchange_name, self._quote_currencies['default'])

    def _validate_symbol(self, symbol: str, exchange_name: str = None) -> bool:
        """Validate trading symbol format"""
        try:
            base, quote = symbol.split('/')
            if not exchange_name:
                return True
            expected_quote = self._get_exchange_quote_currency(exchange_name)
            return quote == expected_quote
        except ValueError:
            return False

    def _get_equivalent_symbol(self, symbol: str, target_exchange: str) -> str:
        """Convert symbol to the appropriate quote currency for the target exchange"""
        try:
            base, _ = symbol.split('/')
            quote = self._get_exchange_quote_currency(target_exchange)
            return f"{base}/{quote}"
        except ValueError:
            return symbol

    def _calculate_profit(self, buy_price: Decimal, sell_price: Decimal) -> Decimal:
        """Calculate potential profit percentage"""
        try:
            return (sell_price - buy_price) / buy_price
        except (TypeError, ZeroDivisionError):
            return Decimal('0')

    def _check_price_deviation(self, prices: List[Decimal]) -> bool:
        """Check if price deviation is within acceptable range"""
        if not prices or len(prices) < 2:
            return False
        
        try:
            prices = [Decimal(str(p)) for p in prices]
            mean_price = sum(prices) / len(prices)
            max_deviation = max(abs(price - mean_price) / mean_price for price in prices)
            return max_deviation <= self.max_price_deviation
        except (TypeError, ZeroDivisionError, ValueError):
            return False

    async def scan_opportunities(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Scan for arbitrage opportunities across exchanges"""
        opportunities = []
        scan_timestamp = get_utc_now()  # Single timestamp for entire scan
        
        try:
            for symbol in symbols:
                # Get tickers from all exchanges
                exchange_prices = {}
                for exchange in self.exchanges:
                    try:
                        # Convert symbol to exchange-specific format
                        exchange_symbol = self._get_equivalent_symbol(symbol, exchange.name)
                        ticker = await exchange.get_ticker(exchange_symbol)
                        if ticker and 'bid' in ticker and 'ask' in ticker:
                            exchange_prices[exchange.name] = {
                                'bid': Decimal(str(ticker['bid'])),
                                'ask': Decimal(str(ticker['ask']))
                            }
                    except Exception as e:
                        logging.error(f"Error getting ticker from {exchange.name}: {str(e)}")
                        continue

                if len(exchange_prices) < 2:
                    continue

                # Check price deviation
                all_prices = [price['bid'] for price in exchange_prices.values()]
                all_prices.extend([price['ask'] for price in exchange_prices.values()])
                
                if not self._check_price_deviation(all_prices):
                    logging.warning(f"Price deviation too high for {symbol}")
                    continue

                # Find arbitrage opportunities
                for buy_ex, buy_prices in exchange_prices.items():
                    for sell_ex, sell_prices in exchange_prices.items():
                        if buy_ex != sell_ex:
                            buy_price = buy_prices['ask']  # Price to buy at
                            sell_price = sell_prices['bid']  # Price to sell at
                            profit_pct = self._calculate_profit(buy_price, sell_price)
                            
                            if profit_pct >= self.min_profit_threshold:
                                opportunity = {
                                    'symbol': symbol,
                                    'buy_exchange': buy_ex,
                                    'sell_exchange': sell_ex,
                                    'buy_price': float(buy_price),
                                    'sell_price': float(sell_price),
                                    'profit_pct': float(profit_pct),
                                    'timestamp': format_timestamp(scan_timestamp)
                                }
                                opportunities.append(opportunity)

        except Exception as e:
            logging.error(f"Error scanning opportunities: {str(e)}")

        return sorted(opportunities, key=lambda x: x['profit_pct'], reverse=True)
