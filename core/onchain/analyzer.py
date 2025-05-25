import ccxt
from web3 import Web3

class OnChainAnalyzer:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('WEB3_PROVIDER')))
        self.cex = ccxt.binance()
    
    def get_combined_flow(self, symbol):
        # Analyse CEX/DEX
        cex_flow = self.cex.fetch_ohlcv(symbol, '1h')
        dex_flow = self.w3.eth.get_blocks('latest', 100)
        return self._merge_flows(cex_flow, dex_flow)

def detect_whale_activity(symbol, threshold=100000):
    """Détection des grosses transactions en temps réel"""
    blocks = self.w3.eth.get_blocks('latest', 50)
    large_txs = [
        tx for tx in blocks 
        if tx['value'] > threshold and tx['token'] == symbol
    ]
    return {
        'count': len(large_txs),
        'total_volume': sum(tx['value'] for tx in large_txs),
        'direction': 'buy' if large_txs[0]['to'] == 'exchange' else 'sell'
    }

def get_cex_dex_arbitrage(symbol):
    """Détection d'opportunités d'arbitrage"""
    cex_price = self.cex.fetch_ticker(symbol)['last']
    dex_price = self.w3.eth.get_token_price(symbol)
    return {
        'cex_price': cex_price,
        'dex_price': dex_price,
        'spread': abs(cex_price - dex_price) / cex_price * 100
    }
