from datetime import datetime

import numpy as np

from modules.utils.advanced_logger import AdvancedLogger


class SignalProcessor:
    def __init__(self):
        self.logger = AdvancedLogger()
        self.signals = []

    def add_signal(self, pair, signal_type, strength):
        """Ajoute un signal au processeur"""
        signal = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "type": signal_type,
            "strength": np.clip(strength, 0, 1),
            "confirmed": False,
        }
        self.signals.append(signal)
        self.logger.log(f"New signal: {pair} {signal_type} ({strength:.2f})")
        return signal

    def confirm_signal(self, signal):
        """Confirme un signal existant"""
        signal["confirmed"] = True
        self.logger.log(f"Signal confirmed: {signal['pair']}", notify=True)
