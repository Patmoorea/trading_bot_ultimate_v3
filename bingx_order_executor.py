from datetime import datetime, timezone

class BingXOrderExecutor:
    def __init__(self, bingx_exchange):
        self.bingx = bingx_exchange

    async def short_order(self, symbol: str, amount: float, leverage: int = 3):
        """
        Ouvre une position short (vendeuse) sur BingX Futures
        symbol: ex 'BTC/USDT:USDT'
        amount: taille du contrat (en coin, pas en USDT)
        """
        try:
            if not self.bingx._initialized:
                await self.bingx.initialize()
            await self.bingx.set_leverage(symbol, leverage, position_side="SHORT")
            order = await self.bingx.create_order(
                symbol=symbol,
                order_type="market",
                side="sell",
                amount=str(amount),
                params={"positionSide": "SHORT"},
            )
            return {
                "status": "completed",
                "order_id": order.get("id"),
                "filled_amount": order.get("amount"),
                "avg_price": order.get("average"),
                "timestamp": datetime.now(timezone.utc),
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now(timezone.utc),
            }
