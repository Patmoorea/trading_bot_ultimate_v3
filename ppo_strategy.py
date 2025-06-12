
# === [Ajout progressif] Ajustement des signaux IA avec biais de sentiment global ===
try:
    from decision_sentiment_adjustment import adjust_signals_with_sentiment

    # Exemple générique : si le modèle PPO donne deux scores
    raw_buy, raw_sell = model.predict(observation)

    # Ajustement avec le biais de sentiment (sans impacter l'ancien système)
    adjusted_buy, adjusted_sell = adjust_signals_with_sentiment(raw_buy, raw_sell)

    # Utilisation dans la logique de décision (ou simplement affichage test)
    print(f"[Ajusté Sentiment] Achat : {adjusted_buy:.4f}, Vente : {adjusted_sell:.4f}")

except Exception as e:
    print(f"[⚠️ ERREUR AJUSTEMENT SENTIMENT] {e}")
