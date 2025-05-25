import time

import ccxt


def fetch_real_time():
    exchange = ccxt.binance()
    while True:
        try:
            btc = exchange.fetch_ticker("BTC/USDC")
            print(f"{time.ctime()} | BTC: {btc['last']}")
            time.sleep(10)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    fetch_real_time()
