
class AdaptiveStopLoss:
    def __init__(self):
        self.volatility_window = 20
    
    def calculate(self, symbol):
        candles = get_historical_data(symbol, '1h', self.volatility_window)
        atr = calculate_atr(candles)
        return atr * 1.5  # Multiplicateur de volatilité
