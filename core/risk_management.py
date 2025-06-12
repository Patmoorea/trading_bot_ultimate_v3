
def adaptive_drawdown(current_pnl):
    # Nouvelle logique dynamique
    base_dd = 0.05
    volatility_adjustment = get_market_volatility() * 0.01
    return max(0.02, base_dd - volatility_adjustment)

def dynamic_position_sizing(volatility, confidence):
    """Taille de position adaptative basée sur la volatilité et la confiance IA"""
    base_size = 0.05  # 5% par défaut
    adj_size = base_size * (1 - volatility) * confidence
    return min(max(adj_size, 0.01), 0.1)  # Entre 1% et 10%

def circuit_breaker(market_conditions):
    """Gestion des krachs marché"""
    if market_conditions['volatility'] > 0.15:
        return {
            'action': 'reduce_positions',
            'message': 'High volatility detected',
            'new_max_drawdown': 0.01
        }
    return {'action': 'normal'}
