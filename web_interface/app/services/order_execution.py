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
        
    async def execute_order(self, 
                          symbol: str, 
                          side: str, 
                          amount: float, 
                          orderbook: Dict,
                          market_data: Dict) -> Dict:
        try:
            # Vérifier que nous sommes en mode achat uniquement
            if side.upper() != "BUY":
                return {
                    "status": "rejected",
                    "reason": "Only BUY orders are allowed",
                    "timestamp": datetime.now(timezone.utc)
                }

            # Vérifier que nous utilisons USDC
            if not symbol.endswith('USDC'):
                return {
                    "status": "rejected",
                    "reason": "Only USDC pairs are allowed",
                    "timestamp": datetime.now(timezone.utc)
                }

            # Optimisation de l'exécution
            execution_plan = self._create_execution_plan(amount, orderbook, market_data)
            
            if not execution_plan["valid"]:
                return {
                    "status": "rejected",
                    "reason": execution_plan["reason"],
                    "timestamp": datetime.now(timezone.utc)
                }

            # Exécution de l'ordre avec protection anti-snipe
            order_result = await self._execute_with_protection(
                symbol, 
                side, 
                execution_plan,
                market_data
            )

            # Mise à jour de l'historique du slippage
            if order_result["status"] == "completed":
                self.slippage_history.append(order_result["slippage"])

            return order_result

        except Exception as e:
            self.logger.error(f"Order execution error: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now(timezone.utc)
            }

    def _create_execution_plan(self, 
                             amount: float, 
                             orderbook: Dict,
                             market_data: Dict) -> Dict:
        try:
            # Analyse de la liquidité
            liquidity = self._analyze_liquidity(orderbook)
            
            # Calcul du prix moyen d'exécution estimé
            avg_price = self._calculate_avg_execution_price(amount, orderbook["asks"])
            
            # Vérification de la profondeur du marché
            if liquidity["depth"] < amount * 2:
                return {
                    "valid": False,
                    "reason": "Insufficient market depth"
                }

            # Création des ordres iceberg si nécessaire
            if amount > liquidity["avg_trade_size"] * 3:
                chunks = self._split_into_iceberg_orders(amount, liquidity)
            else:
                chunks = [amount]

            return {
                "valid": True,
                "chunks": chunks,
                "avg_price": avg_price,
                "liquidity_score": liquidity["score"],
                "estimated_slippage": liquidity["estimated_slippage"]
            }

        except Exception as e:
            self.logger.error(f"Error creating execution plan: {e}")
            return {"valid": False, "reason": str(e)}

    async def _execute_with_protection(self,
                                    symbol: str,
                                    side: str,
                                    execution_plan: Dict,
                                    market_data: Dict) -> Dict:
        try:
            orders_completed = []
            total_filled = Decimal('0')
            avg_price = Decimal('0')
            
            for chunk in execution_plan["chunks"]:
                # Protection anti-snipe
                if self._detect_adverse_price_movement(market_data):
                    await asyncio.sleep(2)  # Attente tactique
                
                # Exécution du chunk avec monitoring en temps réel
                chunk_result = await self._execute_chunk(
                    symbol,
                    side,
                    chunk,
                    execution_plan["avg_price"]
                )
                
                if chunk_result["status"] == "filled":
                    orders_completed.append(chunk_result)
                    total_filled += Decimal(str(chunk_result["filled_amount"]))
                    avg_price += (Decimal(str(chunk_result["filled_amount"])) * 
                                Decimal(str(chunk_result["filled_price"])))
                else:
                    # Gestion des erreurs d'exécution
                    return {
                        "status": "partial",
                        "filled_amount": float(total_filled),
                        "reason": chunk_result["reason"]
                    }

            # Calcul du prix moyen et du slippage
            if total_filled > 0:
                avg_price = avg_price / total_filled
                slippage = (float(avg_price) / execution_plan["avg_price"]) - 1
            else:
                slippage = 0

            return {
                "status": "completed",
                "filled_amount": float(total_filled),
                "avg_price": float(avg_price),
                "slippage": slippage,
                "orders": orders_completed,
                "timestamp": datetime.now(timezone.utc)
            }

        except Exception as e:
            self.logger.error(f"Protected execution error: {e}")
            return {"status": "error", "reason": str(e)}

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
                "estimated_slippage": estimated_slippage
            }
        except Exception as e:
            self.logger.error(f"Liquidity analysis error: {e}")
            return {
                "depth": 0,
                "spread": 999,
                "avg_trade_size": 0,
                "score": 0,
                "estimated_slippage": 999
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
                "successful_orders": len([s for s in self.slippage_history if s <= 0.001]),
                "total_orders": len(self.slippage_history)
            }
        except Exception as e:
            self.logger.error(f"Stats calculation error: {e}")
            return {}
