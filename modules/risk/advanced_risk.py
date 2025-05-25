from datetime import timedelta

from modules.utils.advanced_logger import AdvancedLogger


class AdvancedRiskManager:
    def __init__(self):
        self.logger = AdvancedLogger()
        self.max_daily_loss = 0.02  # 2%
        self.positions = []

    def calculate_position_size(self, balance, risk_per_trade):
        """Calcule la taille de position basée sur le risque"""
        size = balance * risk_per_trade
        self.logger.log(f"Calculated position size: {size:.2f}")
        return size

    def check_daily_drawdown(self, current_balance, initial_balance):
        """Vérifie le drawdown quotidien"""
        loss = (initial_balance - current_balance) / initial_balance
        if loss >= self.max_daily_loss:
            self.logger.log("Daily loss limit reached!", notify=True)
            return False
        return True

    def calculate_kelly_position(self, win_prob, win_loss_ratio):
        """Calcule la position selon le critère de Kelly"""
        kelly_fraction = win_prob - (1 - win_prob) / win_loss_ratio
        return max(kelly_fraction, 0)  # Évite les valeurs négatives
