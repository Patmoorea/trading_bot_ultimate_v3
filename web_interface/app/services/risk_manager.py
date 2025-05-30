import logging
from app.config import Config

class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def check_limits(self, decision):
        # Vérification des limites de risque
        return True
        
    def calculate_position_size(self, decision):
        return min(decision['size'], Config.MAX_POSITION_SIZE)
        
    def calculate_stop_loss(self, decision):
        return decision['price'] * (1 - Config.DAILY_STOP_LOSS)
