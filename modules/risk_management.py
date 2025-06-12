def calculate_dynamic_sl(volatility):
    base_sl = 0.02  # 2% par défaut
    adjusted_sl = base_sl * (volatility / 0.05)  # 5% de vol de référence
    return min(adjusted_sl, 0.05)  # Max 5%
