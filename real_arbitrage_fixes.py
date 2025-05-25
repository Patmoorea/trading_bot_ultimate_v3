import time
from typing import Dict, List, Tuple
import ccxt
import logging
from ..base import BaseStrategy

class USDCArbitrage(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_spread = config.get('min_spread', 0.002)
        self.exchanges = self._init_exchanges(config.get('exchanges', ['binance', 'gateio']))
        self.logger = logging.getLogger(__name__)
        self.timeout = config.get('timeout', 10000)  # 10 secondes par défaut

    def _safe_fetch_order_book(self, exchange, symbol, retries=3):
        """Version robuste de fetch_order_book"""
        for attempt in range(retries):
            try:
                return exchange.fetch_order_book(symbol, {'timeout': self.timeout})
            except Exception as e:
                if attempt == retries - 1:
                    self.logger.error(f"Échec après {retries} tentatives: {str(e)}")
                    raise
                time.sleep(1)

    def scan_all_pairs(self) -> Dict[str, float]:
        """Scan sécurisé des paires USDC"""
        opportunities = {}
        for name, exchange in self.exchanges.items():
            try:
                markets = exchange.load_markets()
                for symbol in markets:
                    if symbol.endswith('/USDC') and markets[symbol]['active']:
                        try:
                            order_book = self._safe_fetch_order_book(exchange, symbol)
                            bid = order_book['bids'][0][0] if order_book['bids'] else 0
                            ask = order_book['asks'][0][0] if order_book['asks'] else 0
                            if bid and ask:
                                spread = (ask - bid) / ask
                                opportunities[f"{name}:{symbol}"] = spread
                        except Exception as e:
                            self.logger.warning(f"Erreur sur {symbol}: {str(e)}")
                            continue
            except Exception as e:
                self.logger.error(f"Erreur sur l'exchange {name}: {str(e)}")
                continue
        return opportunities
