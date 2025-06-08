import ccxt
import json
import time

EXCHANGES = ['binance', 'kraken', 'coinbasepro', 'huobi', 'okx']

def fetch_sample_data():
    results = {}
    for name in EXCHANGES:
        try:
            exchange = getattr(ccxt, name)()
            markets = exchange.load_markets()
            usdc_pairs = [s for s in markets if s.endswith('/USDC') and markets[s]['active']]
            
            sample = {}
            for pair in usdc_pairs[:3]:  # Test sur 3 paires max
                try:
                    # Récupération des données brutes
                    order_book = exchange.fetch_order_book(pair)
                    ticker = exchange.fetch_ticker(pair)
                    sample[pair] = {
                        'order_book_bids': order_book['bids'][0] if order_book['bids'] else None,
                        'order_book_asks': order_book['asks'][0] if order_book['asks'] else None,
                        'ticker': ticker
                    }
                except Exception as e:
                    sample[pair] = f"Error: {str(e)}"
                time.sleep(0.5)  # Respect rate limits
            
            results[name] = sample
        except Exception as e:
            results[name] = f"Init Error: {str(e)}"
    
    with open('exchange_samples.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Données sauvegardées dans exchange_samples.json")

if __name__ == '__main__':
    fetch_sample_data()
