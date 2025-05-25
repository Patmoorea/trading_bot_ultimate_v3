import os
import asyncio
from decimal import Decimal
from ccxt.async_support import binance
from dotenv import load_dotenv

load_dotenv()

class BinanceConnector:
    def __init__(self):
        self.exchange = binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            },
            'enableRateLimit': True
        })
    
    async def get_order_book(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Récupère le carnet d'ordres de Binance"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol)
            bid = Decimal(str(orderbook['bids'][0][0])) if len(orderbook['bids']) > 0 else Decimal(0)
            ask = Decimal(str(orderbook['asks'][0][0])) if len(orderbook['asks']) > 0 else Decimal('Infinity')
            return bid, ask
        except Exception as e:
            raise Exception(f"Binance error: {str(e)}")
    
    async def create_order(self, symbol: str, side: str, amount: Decimal, price: Decimal = None):
        """Crée un ordre sur Binance"""
        try:
            params = {
                'type': 'market' if not price else 'limit',
                'amount': float(amount),
                'price': float(price) if price else None
            }
            return await self.exchange.create_order(symbol, 'market', side, float(amount), params)
        except Exception as e:
            raise Exception(f"Binance order error: {str(e)}")
    
    async def close(self):
        """Ferme la connexion"""
        await self.exchange.close()
