from typing import Dict, Optional, List
import numpy as np
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal


class SmartOrderExecutor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.executing_orders = {}
        self.slippage_history = []
        self.min_profit = 0.001  # 0.1% minimum profit

    async def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float = None,
        quoteOrderQty: float = None,
        orderbook: Dict = None,
        market_data: Dict = None,
        iceberg: bool = False,
        iceberg_visible_size: float = 0.1,
    ) -> Dict:
        try:
            # Vérifier que nous sommes en mode achat uniquement
            if side.upper() != "BUY":
                return {
                    "status": "rejected",
                    "reason": "Only BUY orders are allowed",
                    "timestamp": datetime.now(timezone.utc),
                }

            # Vérifier que nous utilisons USDC
            if not symbol.endswith("USDC"):
                return {
                    "status": "rejected",
                    "reason": "Only USDC pairs are allowed",
                    "timestamp": datetime.now(timezone.utc),
                }

            # Utiliser quoteOrderQty pour les achats USDC (Binance API)
            exec_amount = quoteOrderQty if quoteOrderQty is not None else amount
            if exec_amount is None or exec_amount <= 0:
                return {
                    "status": "rejected",
                    "reason": "Invalid order amount",
                    "timestamp": datetime.now(timezone.utc),
                }

            # Optimisation de l'exécution
            execution_plan = self._create_execution_plan(
                exec_amount, orderbook, market_data
            )

            if not execution_plan["valid"]:
                return {
                    "status": "rejected",
                    "reason": execution_plan["reason"],
                    "timestamp": datetime.now(timezone.utc),
                }

            # Exécution de l'ordre avec protection anti-snipe
            order_result = await self._execute_with_protection(
                symbol,
                side,
                execution_plan,
                market_data,
                iceberg=iceberg,
                iceberg_visible_size=iceberg_visible_size,
            )

            # Mise à jour de l'historique du slippage
            if order_result["status"] == "completed":
                self.slippage_history.append(order_result.get("slippage", 0.0))

            return order_result

        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now(timezone.utc),
            }

    def _create_execution_plan(self, amount, orderbook, market_data):
        # ... Implémentation de ton plan d'exécution ...
        # Doit retourner {"valid": True, ...} ou {"valid": False, "reason": "..."}
        return {"valid": True, "plan": {"amount": amount}}  # Placeholder

    async def _execute_with_protection(
        self,
        symbol,
        side,
        execution_plan,
        market_data,
        iceberg=False,
        iceberg_visible_size=0.1,
    ):
        try:
            binance_client = market_data.get("binance_client")
            if not binance_client:
                return {
                    "status": "error",
                    "reason": "No binance client provided",
                    "timestamp": datetime.now(timezone.utc),
                }

            api_args = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quoteOrderQty": execution_plan["plan"]["amount"],  # PATCH ici !
            }
            if iceberg:
                api_args["icebergQty"] = iceberg_visible_size

            order_response = await binance_client.create_order(**api_args)
            # Analyse la réponse et construis le résultat
            return {
                "status": "completed",
                "filled_amount": order_response.get(
                    "executedQty", execution_plan["plan"]["amount"]
                ),
                "avg_price": float(
                    order_response.get("fills", [{}])[0].get("price", 0)
                ),
                "slippage": 0.0,  # calculer si besoin
                "timestamp": datetime.now(timezone.utc),
            }
        except Exception as e:
            self.logger.error(f"Order execution error (Binance): {e}")
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now(timezone.utc),
            }

    def _analyze_liquidity(self, orderbook: Dict) -> Dict:
        try:
            bids = np.array(orderbook["bids"])
            asks = np.array(orderbook["asks"])

            depth = np.sum(bids[:, 1]) + np.sum(asks[:, 1])
            spread = (asks[0][0] / bids[0][0]) - 1
            avg_trade_size = np.mean(bids[:, 1])

            # Score de liquidité (0-1)
            liquidity_score = min(1.0, depth / 100000) * (1 - min(1.0, spread * 100))

            # Estimation du slippage basée sur la profondeur du marché
            estimated_slippage = spread * 2  # Estimation conservative

            return {
                "depth": depth,
                "spread": spread,
                "avg_trade_size": avg_trade_size,
                "score": liquidity_score,
                "estimated_slippage": estimated_slippage,
            }
        except Exception as e:
            self.logger.error(f"Liquidity analysis error: {e}")
            return {
                "depth": 0,
                "spread": 999,
                "avg_trade_size": 0,
                "score": 0,
                "estimated_slippage": 999,
            }

    def _detect_adverse_price_movement(self, market_data: Dict) -> bool:
        try:
            recent_prices = market_data.get("recent_trades", [])[-10:]
            if not recent_prices:
                return False

            price_changes = np.diff([trade["price"] for trade in recent_prices])

            # Détection de mouvements suspects
            sudden_moves = np.abs(price_changes) > np.std(price_changes) * 3

            return np.any(sudden_moves)

        except Exception as e:
            self.logger.error(f"Price movement detection error: {e}")
            return True  # Par précaution

    def get_execution_stats(self) -> Dict:
        try:
            return {
                "avg_slippage": np.mean(self.slippage_history[-100:]),
                "max_slippage": np.max(self.slippage_history[-100:]),
                "successful_orders": len(
                    [s for s in self.slippage_history if s <= 0.001]
                ),
                "total_orders": len(self.slippage_history),
            }
        except Exception as e:
            self.logger.error(f"Stats calculation error: {e}")
            return {}
