import logging
from datetime import datetime

from modules.utils.telegram_logger import TelegramLogger


class AdvancedLogger:
    def __init__(self):
        self.telegram = TelegramLogger()
        logging.basicConfig(
            filename="trading.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def log(self, message, level="info", notify=False):
        """Niveaux: debug, info, warning, error, critical"""
        msg = f"[{datetime.now().isoformat()}] {message}"
        getattr(logging, level)(msg)
        if notify:
            self.telegram.log(f"🚨 {msg}")


def log_performance(self, operation, elapsed):
    """Journalisation des métriques de performance"""
    msg = f"⏱ {operation} | {elapsed:.4f}s"
    self.log(msg, level="info", notify=False)
