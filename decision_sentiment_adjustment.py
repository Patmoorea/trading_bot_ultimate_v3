from sentiment_influence import compute_sentiment_bias

def adjust_signals_with_sentiment(buy_score: float, sell_score: float) -> tuple[float, float]:
    """
    Ajuste les signaux d'achat/vente à l'aide du biais de sentiment global.

    Args:
        buy_score (float): Score brut du signal d'achat (ex: probabilité PPO).
        sell_score (float): Score brut du signal de vente.

    Returns:
        tuple: Scores ajustés (achat, vente)
    """
    bias = compute_sentiment_bias()

    # Sécurité : éviter division par 0
    if bias <= 0:
        bias = 0.01

    # Influence positive = plus d’achat / moins de vente
    adjusted_buy = buy_score * bias
    adjusted_sell = sell_score / bias

    return adjusted_buy, adjusted_sell
