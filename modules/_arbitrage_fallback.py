import ccxt

class FallbackArbitrage:
    """Wrapper de la fonction existante en classe"""
    
    @staticmethod
    def calculate():
        return safe_find_arbitrage()

def safe_find_arbitrage():
    try:
        exchange = ccxt.binance()
        btc_usdc = exchange.fetch_order_book('BTC/USDC')
        spread = (btc_usdc['asks'][0][0] - btc_usdc['bids'][0][0]) / btc_usdc['asks'][0][0]
        return {'BTC/USDC': f"{spread*100:.2f}%"}
    except Exception as e:
        return f"Erreur: {str(e)}"
