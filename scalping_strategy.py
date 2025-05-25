
# === ÉVOLUTION : Ajustement des signaux IA avec le sentiment du marché ===
try:
    from decision_sentiment_adjustment import adjust_signals_with_sentiment

    def scalping_decision_with_sentiment(buy_score, sell_score):
        """
        Version évoluée du signal de scalping avec ajustement par sentiment global.
        """
        adjusted_buy, adjusted_sell = adjust_signals_with_sentiment(buy_score, sell_score)
        return adjusted_buy, adjusted_sell
except Exception as e:
    print(f"[ERREUR] Ajustement sentiment échoué dans scalping_strategy : {e}")
