class MultiExchangeArbitrage:
    def __init__(self):
        self.exchanges = ['binance', 'OKX', 'GATEIO', 'BINGX', 'BLOFIN']
        
    def get_best_spread(self):
        """Retourne un exemple de spread pour tester"""
        return {
            'exchange_pair': ('binance', 'okx', 'gateio', 'bingx', 'blofin'),
            'spread': 0.45,
            'bid': 104200.0,
            'ask': 103800.0
        }
