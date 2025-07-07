def breakout_strategy(df, lookback=20, take_profit=2.0):
    """
    Stratégie Breakout simple : signal 1 si close casse le plus haut du lookback, -1 sinon.
    """
    high = df["high"].rolling(lookback).max()
    signals = (df["close"] > high.shift(1)).astype(int)
    signals = signals.where(signals == 1, -1)
    signals.index = df.index
    return signals
