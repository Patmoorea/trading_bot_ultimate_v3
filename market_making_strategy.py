
# === ÉVOLUTION : Ajustement des signaux IA avec le sentiment du marché ===
try:
    from decision_sentiment_adjustment import adjust_signals_with_sentiment

    def market_making_decision_with_sentiment(buy_score, sell_score):
        """
        Version évoluée du signal de market making avec sentiment.
        """
        adjusted_buy, adjusted_sell = adjust_signals_with_sentiment(buy_score, sell_score)
        return adjusted_buy, adjusted_sell
except Exception as e:
    print(f"[ERREUR] Ajustement sentiment échoué dans market_making_strategy : {e}")
