from decimal import Decimal, ROUND_DOWN

class PositionManager:
    def __init__(self, account_balance, max_risk_per_trade=0.02):
        self.account_balance = Decimal(str(account_balance))
        self.max_risk_per_trade = Decimal(str(max_risk_per_trade))

    def calculate_position_size(self, entry_price, stop_loss, leverage=1):
        """
        Calcule la taille de position optimale selon le risque défini
        """
        risk_amount = self.account_balance * self.max_risk_per_trade
        price_diff = abs(Decimal(str(entry_price)) - Decimal(str(stop_loss)))
        position_size = (risk_amount / price_diff) * Decimal(str(leverage))
        
        return position_size.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)

    def validate_trade(self, position_size, current_positions):
        """
        Vérifie si le trade respecte les règles de gestion du risque
        """
        total_exposure = sum(pos['size'] for pos in current_positions)
        new_exposure = total_exposure + position_size
        
        max_exposure = self.account_balance * Decimal('2.5')  # 250% max exposure
        return new_exposure <= max_exposure

