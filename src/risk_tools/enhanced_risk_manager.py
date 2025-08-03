class EnhancedRiskManager:
    def __init__(self):
        self.max_drawdown_limit = 0.15
        self.position_limits = {
            "max_per_trade": 0.05,
            "max_total_exposure": 0.25
        }
        self.min_confidence = 0.8
        self.correlation_threshold = 0.7

    def calculate_position_size(self, equity, confidence, volatility, correlation):
        """Calcule la taille optimale de position"""
        if confidence < self.min_confidence:
            return 0
        
        base_size = equity * self.position_limits["max_per_trade"]
        vol_adj = max(0.3, 1 - (volatility * 2))
        corr_adj = max(0.3, 1 - correlation)
        
        return min(
            base_size * vol_adj * corr_adj,
            equity * self.position_limits["max_per_trade"]
        )

    def validate_trade(self, signals):
        """Vérifie si le trade respecte tous les critères"""
        score = 0
        
        # Vérification tendance
        if signals.get("trend_score", 0) > 0.7:
            score += 2
            
        # Vérification volume
        if signals.get("volume_score", 0) > 0.6:
            score += 2
            
        # Vérification momentum
        if signals.get("momentum_score", 0) > 0.6:
            score += 1
            
        # Vérification support/résistance
        if signals.get("sr_score", 0) > 0.7:
            score += 1
            
        return score >= 5

    def check_exposure_limit(self, current_positions, new_position_size):
        """Vérifie si nouvelle position respecte limite exposition"""
        total_exposure = sum(pos["size"] for pos in current_positions.values())
        return (total_exposure + new_position_size) <= self.position_limits["max_total_exposure"]