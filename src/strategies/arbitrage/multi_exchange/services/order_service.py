"""
Service de gestion des ordres - Created: 2025-05-17 23:18:55
@author: Patmoorea
"""
from typing import Dict, List
import ccxt
from datetime import datetime

class OrderService:
    def __init__(self, exchanges: Dict[str, ccxt.Exchange]):
        self.exchanges = exchanges
        self.last_update = "2025-05-17 23:18:55"

    async def place_orders(self, orders: List[Dict]) -> List[Dict]:
        results = []
        for order in orders:
            try:
                exchange = self.exchanges[order['exchange']]
                result = await exchange.create_order(
                    symbol=order['symbol'],
                    type=order['type'],
                    side=order['side'],
                    amount=order['amount'],
                    price=order.get('price')
                )
                results.append({
                    'success': True,
                    'order': result,
                    'exchange': order['exchange'],
                    'timestamp': datetime.utcnow()
                })
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'exchange': order['exchange'],
                    'timestamp': datetime.utcnow()
                })
        return results
