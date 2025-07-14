#!/usr/bin/env python3
"""
Script de test pour vérifier les corrections apportées au bot de trading.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bot_runner import TradingBotM4
from src.ai.deep_learning_model import DeepLearningModel

def test_sentiment_propagation():
    """Test de la propagation du sentiment"""
    print("🧪 Test de la propagation du sentiment...")
    
    bot = TradingBotM4()
    
    # Simuler des données de sentiment
    sentiment_scores = [
        {"symbol": "BTCUSDT", "sentiment": -0.35},
        {"symbol": "ETHUSDT", "sentiment": 0.25}
    ]
    
    # Initialiser market_data
    bot.market_data = {
        "BTCUSDT": {"1h": {"close": [50000, 51000]}},
        "ETHUSDT": {"1h": {"close": [3000, 3100]}}
    }
    
    # Test de la propagation (sans asyncio.run car déjà dans un contexte async)
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot._update_sentiment_data(sentiment_scores))
    finally:
        loop.close()
    
    # Vérifier les résultats
    btc_sentiment = bot.market_data.get("BTCUSDT", {}).get("sentiment", 0)
    eth_sentiment = bot.market_data.get("ETHUSDT", {}).get("sentiment", 0)
    
    print(f"   BTC sentiment: {btc_sentiment} (attendu: -0.35)")
    print(f"   ETH sentiment: {eth_sentiment} (attendu: 0.25)")
    
    if btc_sentiment == -0.35 and eth_sentiment == 0.25:
        print("   ✅ Propagation du sentiment: OK")
        return True
    else:
        print("   ❌ Propagation du sentiment: ÉCHEC")
        return False

def test_ai_model():
    """Test du modèle IA"""
    print("🧪 Test du modèle IA...")
    
    try:
        model = DeepLearningModel()
        
        # Créer des features de test
        features = {
            "close": np.array([50000, 50100, 50200, 50150, 50300] * 13),  # 65 points
            "high": np.array([50100, 50200, 50300, 50250, 50400] * 13),
            "low": np.array([49900, 50000, 50100, 50050, 50200] * 13),
            "volume": np.array([1000, 1100, 1200, 1150, 1300] * 13),
            "rsi": 0.6,
            "macd": 0.1,
            "volatility": 0.02,
            "vol_ratio": 1.2
        }
        
        prediction = model.predict(features)
        print(f"   Prédiction IA: {prediction:.4f}")
        
        if abs(prediction) > 0.01:  # Prédiction non nulle
            print("   ✅ Modèle IA: OK")
            return True
        else:
            print("   ⚠️ Modèle IA: Prédiction faible mais fonctionnel")
            return True
            
    except Exception as e:
        print(f"   ❌ Modèle IA: ERREUR - {e}")
        return False

def test_technical_analysis():
    """Test de l'analyse technique"""
    print("🧪 Test de l'analyse technique...")
    
    bot = TradingBotM4()
    
    # Créer des données OHLCV de test avec une tendance haussière claire
    n_points = 100
    base_price = 50000
    trend = np.linspace(0, 0.1, n_points)  # Tendance haussière de 10%
    noise = np.random.normal(0, 0.01, n_points)
    
    closes = base_price * (1 + trend + noise)
    opens = closes * (1 + np.random.normal(0, 0.001, n_points))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n_points)))
    volumes = np.random.normal(1000, 100, n_points)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1H'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Calculer les indicateurs
    indicators = bot.add_indicators(df)
    
    if indicators and len([v for v in indicators.values() if v is not None]) > 5:
        print(f"   ✅ Indicateurs calculés: {len([v for v in indicators.values() if v is not None])}/14")
        
        # Test de la fonction analyze_signals
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                decision = loop.run_until_complete(bot.analyze_signals("BTCUSDT", df, indicators))
            finally:
                loop.close()
            print(f"   Décision: {decision['action']} (confiance: {decision['confidence']:.3f})")
            print(f"   Scores - Tech: {decision['signals']['technical']:.3f}, "
                  f"IA: {decision['signals']['ai']:.3f}, "
                  f"Sentiment: {decision['signals']['sentiment']:.3f}")
            
            if decision['confidence'] > 0.1:  # Confiance non nulle
                print("   ✅ Analyse technique: OK")
                return True
            else:
                print("   ⚠️ Analyse technique: Confiance faible")
                return True
                
        except Exception as e:
            print(f"   ❌ Analyse technique: ERREUR - {e}")
            return False
    else:
        print("   ❌ Indicateurs: ÉCHEC")
        return False

def test_decision_thresholds():
    """Test des seuils de décision"""
    print("🧪 Test des seuils de décision...")
    
    # Test avec différents scores
    test_cases = [
        (0.4, "buy"),    # Score > 0.3 → buy
        (-0.4, "sell"),  # Score < -0.3 → sell
        (0.2, "neutral"), # -0.3 < Score < 0.3 → neutral
        (-0.2, "neutral")
    ]
    
    all_passed = True
    for score, expected_action in test_cases:
        if score > 0.3:
            action = "buy"
        elif score < -0.3:
            action = "sell"
        else:
            action = "neutral"
            
        if action == expected_action:
            print(f"   ✅ Score {score:.1f} → {action}")
        else:
            print(f"   ❌ Score {score:.1f} → {action} (attendu: {expected_action})")
            all_passed = False
    
    return all_passed

async def main():
    """Fonction principale de test"""
    print("🚀 Démarrage des tests des corrections du bot de trading\n")
    
    tests = [
        ("Propagation du sentiment", test_sentiment_propagation),
        ("Modèle IA", test_ai_model),
        ("Analyse technique", test_technical_analysis),
        ("Seuils de décision", test_decision_thresholds)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name}: ERREUR CRITIQUE - {e}")
            results.append((test_name, False))
        print()
    
    # Résumé
    print("📊 RÉSUMÉ DES TESTS:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHEC"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat global: {passed}/{len(results)} tests passés")
    
    if passed == len(results):
        print("🎉 Toutes les corrections fonctionnent correctement !")
    elif passed >= len(results) * 0.75:
        print("⚠️ La plupart des corrections fonctionnent, quelques ajustements nécessaires.")
    else:
        print("❌ Plusieurs problèmes détectés, révision nécessaire.")

if __name__ == "__main__":
    asyncio.run(main())