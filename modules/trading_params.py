import os

from dotenv import load_dotenv

load_dotenv()


class TradingConfig:
    @property
    def stop_loss(self):
        return float(os.getenv("STOP_LOSS", 0.05))

    @property
    def take_profit(self):
        return float(os.getenv("TAKE_PROFIT", 0.15))

    @property
    def max_position(self):
        return float(os.getenv("MAX_POSITION", 0.1))
