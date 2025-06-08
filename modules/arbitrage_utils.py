def get_active_usdc_pairs():
    """Liste les paires USDC avec volume"""
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    return [
        p for p in markets 
        if p.endswith('/USDC') 
        and markets[p]['active'] 
        and markets[p]['percentage'] > 0.5  # Spread minimum naturel
    ]

import ccxt  # Import manquant ajouté
