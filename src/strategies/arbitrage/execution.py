import logging
from typing import Dict
from .core import ArbitrageOpportunity


class ArbitrageExecutor:
    def __init__(self, exchanges: Dict[str, object]):
        self.exchanges = exchanges
        self.logger = logging.getLogger(__name__)

    async def execute(
        self, opportunity: ArbitrageOpportunity, max_slippage=0.1, timeout=5
    ):
        """
        Exécute une opération d'arbitrage (ASYNC)
        """
        try:
            exch_b = self.exchanges[opportunity.exchange_b]
            exch_a = self.exchanges[opportunity.exchange_a]

            # Création de l'ordre buy (async)
            buy_order = await exch_b.create_order(
                symbol=opportunity.pair,
                type="limit",
                side="buy",
                amount=opportunity.volume,
                price=opportunity.ask_b,
            )

            # Création de l'ordre sell (async)
            sell_order = await exch_a.create_order(
                symbol=opportunity.pair,
                type="limit",
                side="sell",
                amount=opportunity.volume,
                price=opportunity.bid_a,
            )

            realized_profit = (
                opportunity.bid_a - opportunity.ask_b
            ) * opportunity.volume
            return {
                "success": True,
                "buy_order": buy_order,
                "sell_order": sell_order,
                "realized_profit": realized_profit,
                "route": f"{opportunity.exchange_b} → {opportunity.exchange_a}",
            }
        except Exception as e:
            self.logger.error(f"Erreur d'exécution arbitrage: {str(e)}")
            await self.cancel_all_orders(opportunity)
            return {"success": False, "error": str(e)}

    async def cancel_all_orders(self, opportunity):
        """Annule tous les ordres en cas d'erreur (ASYNC)"""
        for exch_name in [opportunity.exchange_a, opportunity.exchange_b]:
            exch = self.exchanges[exch_name]
            try:
                await exch.cancel_all_orders(opportunity.pair)
            except Exception as e:
                self.logger.error(
                    f"Erreur lors de l'annulation des ordres sur {exch_name}: {e}"
                )
