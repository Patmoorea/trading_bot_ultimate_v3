from decimal import Decimal
from typing import Dict, Optional
from ..base_exchange import BaseExchange
import ccxt
import logging

class BinanceClient(BaseExchange):
    def __init__(self, api_key: str, api_secret: str):  # Changé de 'secret' à 'api_secret'
        super().__init__(api_key, api_secret)  # Aussi mis à jour ici
        self.logger = logging.getLogger(__name__)
        self._initialize_exchange()

    def _initialize_exchange(self):
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,  # Utilisez self.api_secret ici
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })

    def get_ticker(self, symbol: str) -> Dict:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'bid': Decimal(str(ticker['bid'])),
                'ask': Decimal(str(ticker['ask'])),
                'last': Decimal(str(ticker['last'])),
                'volume': Decimal(str(ticker['baseVolume']))
            }
        except Exception as e:
            self.logger.error(f"Erreur Binance get_ticker: {str(e)}")
            raise

    def get_balance(self) -> Dict:
        try:
            balance = self.exchange.fetch_balance()
            if not balance or 'total' not in balance:
                raise ValueError("Balance invalide ou vide")

            return {
                currency: {
                    'free': Decimal(str(info.get('free', 0))),
                    'used': Decimal(str(info.get('used', 0))),
                    'total': Decimal(str(info.get('total', 0)))
                }
                for currency, info in balance.get('total', {}).items()
                if isinstance(info, dict) and info.get('total', 0) > 0
            }
        except Exception as e:
            self.logger.error(f"Erreur Binance get_balance: {str(e)}")
            raise

    def place_order(self, symbol: str, side: str, amount: Decimal, 
                   price: Optional[Decimal] = None) -> Dict:
        try:
            order_type = 'market' if price is None else 'limit'
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=float(amount),
                price=float(price) if price else None
            )
            return order
        except Exception as e:
            self.logger.error(f"Erreur Binance place_order: {str(e)}")
            raise

    def get_order_book(self, symbol: str) -> Dict:
        try:
            book = self.exchange.fetch_order_book(symbol)
            return {
                'bids': [[Decimal(str(price)), Decimal(str(amount))] 
                        for price, amount in book['bids']],
                'asks': [[Decimal(str(price)), Decimal(str(amount))] 
                        for price, amount in book['asks']]
            }
        except Exception as e:
            self.logger.error(f"Erreur Binance get_order_book: {str(e)}")
            raise
