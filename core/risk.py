class RiskManager:
    def __init__(self, max_drawdown=0.05):
        self.max_drawdown = max_drawdown
    
    def calculate_max_position(self, capital):
        return capital * (1 - self.max_drawdown)
