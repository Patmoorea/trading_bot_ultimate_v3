import ccxt
import time
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent/'config'/'arbitrage.json'

def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except:
        return {
            "min_spread": 0.0005,
            "target_pairs": ["BTC/USDC", "ETH/USDC"],
            "update_interval": 10,
            "exchanges": {"binance": {"enabled": True}}
        }

def init_exchanges(config):
    exchanges = {}
    for name, params in config['exchanges'].items():
        if params.get('enabled'):
            try:
                exchange = getattr(ccxt, name)({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                # Vérification de la disponibilité des paires USDC
                markets = exchange.load_markets()
                usdc_pairs = [p for p in config['target_pairs'] if p in markets]
                if usdc_pairs:
                    exchanges[name] = exchange
                    print(f"{name} OK - Paires USDC: {', '.join(usdc_pairs)}")
                else:
                    print(f"{name} ignore - Pas de paires USDC disponibles")
            except Exception as e:
                print(f"Erreur {name}: {str(e)}")
    return exchanges

def main():
    config = load_config()
    exchanges = init_exchanges(config)
    
    while True:
        try:
            for pair in config['target_pairs']:
                for name, exchange in exchanges.items():
                    try:
                        orderbook = exchange.fetch_order_book(pair)
                        spread = (orderbook['asks'][0][0] - orderbook['bids'][0][0]) / orderbook['asks'][0][0]
                        if spread > config['min_spread']:
                            print(f"\033[92m{name} {pair}: {spread*100:.4f}%\033[0m")
                        else:
                            print(f"{name} {pair}: {spread*100:.4f}%")
                    except:
                        continue
            time.sleep(config['update_interval'])
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
