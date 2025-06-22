import asyncio

try:
    print(">>> AVANT load_markets (timeout 10s)")
    await asyncio.wait_for(self._exchange.load_markets(), timeout=10)
    print(">>> APRES load_markets")
except Exception as e:
    import traceback
    print("=== EXCEPTION load_markets ===")
    print(traceback.format_exc())
    raise