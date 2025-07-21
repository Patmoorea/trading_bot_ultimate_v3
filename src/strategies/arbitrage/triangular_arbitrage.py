"""
Implémentation de l'arbitrage triangulaire réel et robuste
Version: 2.0.0
"""

from typing import Dict, List, Tuple, Optional
import asyncio
import logging
from datetime import datetime
from .service import MoteurArbitrage


class TriangularArbitrage:
    def __init__(self, config: Dict):
        self.config = config
        self.moteur = MoteurArbitrage(config)
        self.min_profit = config.get("min_profit", 0.001)
        self.fee_threshold = config.get("fee_threshold", 0.003)
        self.min_volume = config.get("min_volume", 100)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        await self.moteur.initialize()

    async def close(self):
        await self.moteur.close()

    async def find_triangular_opportunities(
        self, pairs: List[str] = None
    ) -> List[Dict]:
        if not pairs:
            pairs = self.config.get("pairs", [])
        orderbooks = {}
        # Fetch all orderbooks for all exchanges and pairs
        for pair in pairs:
            for name, exchange in self.moteur.exchanges.items():
                try:
                    ob = await self.moteur.fetch_orderbook(exchange, pair)
                    if ob:
                        if pair not in orderbooks:
                            orderbooks[pair] = {}
                        orderbooks[pair][name] = ob
                except Exception as e:
                    self.logger.warning(
                        f"Erreur fetch orderbook {pair} sur {name}: {str(e)}"
                    )
        opportunities = []
        # Triangular path scan
        for pair_a in pairs:
            for pair_b in pairs:
                if pair_a != pair_b:
                    for pair_c in pairs:
                        if pair_c != pair_a and pair_c != pair_b:
                            opp = await self.calculate_triangular_profit(
                                (pair_a, pair_b, pair_c), orderbooks
                            )
                            if opp and opp["profit_pct"] > self.min_profit:
                                opportunities.append(opp)
        return opportunities

    async def calculate_path_profit(
        self, path: Tuple[str, str, str], exchange: str, orderbooks: Dict
    ) -> Optional[Dict]:
        try:
            volume = self.min_volume
            rates = []
            pairs = []
            for i in range(len(path) - 1):
                base = path[i]
                quote = path[i + 1]
                pair = f"{base}/{quote}"
                inverse_pair = f"{quote}/{base}"
                # Direct or inverse pair
                if pair in orderbooks and exchange in orderbooks[pair]:
                    pairs.append(pair)
                    rates.append(orderbooks[pair][exchange]["asks"][0][0])
                    volume = min(volume, orderbooks[pair][exchange]["asks"][0][1])
                elif (
                    inverse_pair in orderbooks and exchange in orderbooks[inverse_pair]
                ):
                    pairs.append(inverse_pair)
                    rates.append(1 / orderbooks[inverse_pair][exchange]["bids"][0][0])
                    volume = min(
                        volume, orderbooks[inverse_pair][exchange]["bids"][0][1]
                    )
                else:
                    self.logger.warning(
                        f"Pair manquant sur {exchange}: {pair} ou {inverse_pair}"
                    )
                    return None
            # Calcul du profit
            total_rate = 1
            for rate in rates:
                total_rate *= rate
            profit = (1 / total_rate) - 1 - (len(path) * self.fee_threshold)
            return {
                "exchange": exchange,
                "path": path,
                "pairs": pairs,
                "rates": rates,
                "profit_pct": profit * 100,
                "volume_constraint": volume,
            }
        except Exception as e:
            self.logger.error(f"Erreur calcul profit: {str(e)}")
            return None

    async def calculate_triangular_profit(
        self, pairs: Tuple[str, str, str], orderbooks: Dict
    ) -> Optional[Dict]:
        best_profit = None
        for exchange in self.moteur.exchanges:
            profit = await self.calculate_path_profit(pairs, exchange, orderbooks)
            if profit and (
                not best_profit or profit["profit_pct"] > best_profit["profit_pct"]
            ):
                best_profit = profit
        return best_profit
