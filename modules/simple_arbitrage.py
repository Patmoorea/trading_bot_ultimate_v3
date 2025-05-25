import json
import time
from pathlib import Path

import ccxt

CONFIG_PATH = Path(__file__).parent.parent / "config" / "arbitrage.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main():
    config = load_config()
    binance = ccxt.binance({"enableRateLimit": True})

    while True:
        try:
            for pair in config["target_pairs"]:
                orderbook = binance.fetch_order_book(pair)
                ask = orderbook["asks"][0][0]
                bid = orderbook["bids"][0][0]
                spread = (ask - bid) / ask

                if spread > config["min_spread"]:
                    print(
                        f"\033[92m{pair}: Spread {spread*100:.4f}% (ASK: {ask} | BID: {bid})\033[0m"
                    )
                else:
                    print(f"{pair}: Spread trop faible {spread*100:.4f}%")

            time.sleep(config["update_interval"])

        except Exception as e:
            print(f"\033[91mErreur: {str(e)}\033[0m")
            time.sleep(30)


if __name__ == "__main__":
    main()
