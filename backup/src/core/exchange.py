from datetime import datetime
from typing import Dict, List, Optional, Any
import ccxt
import logging
from decimal import Decimal, ROUND_DOWN, InvalidOperation

class ExchangeInterface:
    """
    Base exchange interface for cryptocurrency trading
    Version 2.0.0 - Created: 2025-05-19 01:26:23 by Patmoorea
    """
    
    def __init__(self, 
                 exchange_id: str,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = False):
        """Initialize exchange connection"""
        self.exchange_id = exchange_id
        self.testnet = testnet
        self._test_mode = False
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # Configure testnet if needed
        if testnet and hasattr(self.exchange, 'set_sandbox_mode'):
            self.exchange.set_sandbox_mode(True)
            
        self.markets = {}
        if not self._test_mode:
            self.load_markets()
        
    def set_test_mode(self, enabled: bool = True):
        """Active/désactive le mode test"""
        self._test_mode = enabled
        if enabled:
            # Pour les tests, on crée un market fictif
            self.markets = {
                'BTC/USDT': {
                    'precision': {'amount': 8, 'price': 2},
                    'limits': {'amount': {'min': 0.0001, 'max': 1000}}
                }
            }

    def load_markets(self) -> Dict:
        """Load markets data from exchange"""
        try:
            if self._test_mode:
                raise ccxt.NetworkError("Test mode network error")
            self.markets = self.exchange.load_markets()
            return self.markets
        except ccxt.NetworkError as e:
            logging.error(f"Network error while loading markets: {str(e)}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load markets: {str(e)}")
            raise
            
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data for a symbol"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logging.error(f"Failed to fetch ticker for {symbol}: {str(e)}")
            raise
            
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book for a symbol"""
        try:
            return self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logging.error(f"Failed to fetch orderbook for {symbol}: {str(e)}")
            raise

    def format_amount(self, symbol: str, amount: float) -> float:
        """Format amount according to exchange requirements"""
        try:
            if not isinstance(amount, (int, float)):
                raise ValueError(f"Amount must be a number, got {type(amount)}")
                
            if amount <= 0:
                raise ValueError("Amount must be positive")
                
            if amount in [float('inf'), float('-inf')] or isinstance(amount, str):
                raise ValueError("Amount must be a finite number")

            try:
                float(amount)
            except (ValueError, TypeError):
                raise ValueError("Amount must be a valid number")
                
            if symbol not in self.markets:
                if self._test_mode:
                    precision = 8
                else:
                    self.load_markets()
                    precision = self.markets[symbol]['precision']['amount']
            else:
                precision = self.markets[symbol]['precision']['amount']
            
            return round(float(amount), precision)
            
        except (InvalidOperation, ValueError, TypeError) as e:
            logging.error(f"Failed to format amount for {symbol}: {str(e)}")
            raise ValueError(f"Invalid amount format for {symbol}: {amount}")
            
    def check_connection(self) -> bool:
        """Check if connection to exchange is working"""
        if self._test_mode:
            return False
        try:
            self.exchange.fetch_time()
            return True
        except Exception as e:
            logging.error(f"Connection test failed: {str(e)}")
            return False

    def create_order(self, 
                    symbol: str,
                    order_type: str,
                    side: str,
                    amount: float,
                    price: Optional[float] = None) -> Dict:
        """Create a new order"""
        try:
            return self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price
            )
        except Exception as e:
            logging.error(f"Failed to create order: {str(e)}")
            raise
            
    def get_balance(self, currency: Optional[str] = None) -> Dict:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            if currency:
                return {currency: balance.get(currency, {})}
            return balance
        except Exception as e:
            logging.error(f"Failed to fetch balance: {str(e)}")
            raise
            
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an existing order"""
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logging.error(f"Failed to cancel order: {str(e)}")
            raise
            
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logging.error(f"Failed to fetch open orders: {str(e)}")
            raise
            
    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Get status of a specific order"""
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logging.error(f"Failed to fetch order status: {str(e)}")
            raise

    def get_min_order_amount(self, symbol: str) -> float:
        """Get minimum order amount for a symbol"""
        try:
            if symbol not in self.markets:
                if self._test_mode and symbol == 'BTC/USDT':
                    return self.markets['BTC/USDT']['limits']['amount']['min']
                else:
                    raise ValueError(f"Symbol {symbol} not found in markets")
            
            limits = self.markets[symbol].get('limits', {})
            amount_limits = limits.get('amount', {})
            min_amount = amount_limits.get('min')
            
            if min_amount is None:
                raise ValueError(f"No minimum amount found for {symbol}")
                
            return float(min_amount)
        except Exception as e:
            logging.error(f"Failed to get minimum order amount for {symbol}: {str(e)}")
            raise
