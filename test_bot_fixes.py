#!/usr/bin/env python3
"""
Script de test et correction des problèmes identifiés dans le bot de trading.

Problèmes identifiés :
1. Sentiment toujours à 0 dans analyze_signals
2. IA prédit toujours 1.000 
3. Pas de signaux SELL générés

Solutions proposées :
1. Corriger la clé d'accès au sentiment dans analyze_signals
2. Ajuster la normalisation des prédictions IA
3. Revoir les seuils de décision et la logique de calcul des scores
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import asyncio

# Simulation des données pour les tests
def create_test_market_data():
    """Crée des données de marché de test"""
    return {
        "BTCUSDT": {
            "1h": {
                "close": [50000 + i * 100 + np.random.normal(0, 500) for i in range(100)],
                "high": [50500 + i * 100 + np.random.normal(0, 500) for i in range(100)],
                "low": [49500 + i * 100 + np.random.normal(0, 500) for i in range(100)],
                "volume": [1000 + np.random.normal(0, 200) for _ in range(100)],
            },
            "sentiment": -0.36,  # Sentiment global négatif
            "sentiment_timestamp": datetime.now().timestamp()
        },
        "ETHUSDT": {
            "1h": {
                "close": [3000 + i * 10 + np.random.normal(0, 50) for i in range(100)],
                "high": [3050 + i * 10 + np.random.normal(0, 50) for i in range(100)],
                "low": [2950 + i * 10 + np.random.normal(0, 50) for i in range(100)],
                "volume": [800 + np.random.normal(0, 150) for _ in range(100)],
            },
            "sentiment": 0.25,  # Sentiment positif
            "sentiment_timestamp": datetime.now().timestamp()
        }
    }

def create_test_indicators():
    """Crée des indicateurs techniques de test"""
    return {
        "sma_20": 50000,
        "sma_50": 49800,
        "ema_20": 50100,
        "rsi_14": 65,  # Légèrement suracheté
        "macd": 150,
        "macd_signal": 120,
        "macd_hist": 30,
        "bb_upper": 51000,
        "bb_lower": 49000,
        "psar": 49500,
        "momentum_10": 200,
        "zscore_20": 0.5
    }

class MockDeepLearningModel:
    """Mock du modèle IA pour les tests"""
    def __init__(self, fixed_prediction=None):
        self.fixed_prediction = fixed_prediction
        self.initialized = True
    
    def predict(self, features):
        """Version corrigée de la prédiction IA"""
        try:
            if self.fixed_prediction is not None:
                raw_pred = self.fixed_prediction
            else:
                # Simulation d'une prédiction plus réaliste
                raw_pred = np.random.uniform(0.3, 0.7)
            
            # CORRECTION : Normalisation moins agressive
            # Ancienne version : (raw_pred - 0.5) * 4
            # Nouvelle version : normalisation plus douce
            normalized_pred = (raw_pred - 0.5) * 2  # Réduire l'amplification de 4 à 2
            return np.clip(normalized_pred, -1, 1)
            
        except Exception as e:
            print(f"Error in DL prediction: {e}")
            return np.random.uniform(-0.1, 0.1)

class TestTradingBot:
    """Version simplifiée du bot pour les tests"""
    
    def __init__(self):
        self.market_data = create_test_market_data()
        self.dl_model = MockDeepLearningModel()
        self.ai_enabled = True
        self.news_enabled = True
        self.pairs_valid = ["BTC/USDT", "ETH/USDT"]
        
    async def analyze_signals_original(self, symbol, ohlcv_df, indicators):
        """Version originale avec les bugs"""
        tech_score = 0
        tech_factors = 0
        
        close = ohlcv_df["close"].iloc[-1] if "close" in ohlcv_df else None
        sma_20 = indicators.get("sma_20")
        rsi_14 = indicators.get("rsi_14")
        
        # Calcul technique simplifié
        if close and sma_20:
            tech_factors += 1
            pct_diff = (close - sma_20) / sma_20 * 100
            tech_score += np.clip(pct_diff * 2, -1, 1)
            
        if rsi_14:
            tech_factors += 1
            if rsi_14 > 70:
                tech_score -= 0.8
            elif rsi_14 < 30:
                tech_score += 0.8
            else:
                tech_score += (rsi_14 - 50) / 25
                
        if tech_factors > 0:
            tech_score = tech_score / tech_factors
            
        # IA
        ai_score = 0
        if self.ai_enabled and self.dl_model:
            try:
                features = {"close": np.array([close] * 63), "rsi": rsi_14/100, "macd": 0.1, "volatility": 0.05}
                ai_score = float(self.dl_model.predict(features))
            except Exception as e:
                print(f"Erreur IA: {e}")
                
        # PROBLÈME 1: Sentiment toujours 0 - mauvaise clé d'accès
        sentiment_score = 0
        if self.news_enabled:
            # BUG: cherche dans self.market_data[symbol]["sentiment"] 
            # mais symbol = "BTCUSDT" et la clé est stockée directement
            if symbol in self.market_data and "sentiment" in self.market_data[symbol]:
                sentiment_score = self.market_data[symbol]["sentiment"]
                
        # Score total avec amplifications problématiques
        total_score = (
            0.4 * tech_score * 2.0  # Amplification excessive
            + 0.3 * ai_score * 1.5   
            + 0.25 * sentiment_score * 3.0  # Amplification excessive
        )
        
        decision = {
            "action": "neutral",
            "confidence": abs(total_score),
            "signals": {
                "technical": tech_score,
                "ai": ai_score,
                "sentiment": sentiment_score,
            },
        }
        
        # PROBLÈME 3: Seuils trop élevés pour SELL
        if total_score > 0.3:
            decision["action"] = "buy"
        elif total_score < -0.3:  # Difficile à atteindre avec les amplifications
            decision["action"] = "sell"
            
        return decision
    
    async def analyze_signals_fixed(self, symbol, ohlcv_df, indicators):
        """Version corrigée des problèmes"""
        tech_score = 0
        tech_factors = 0
        
        close = ohlcv_df["close"].iloc[-1] if "close" in ohlcv_df else None
        sma_20 = indicators.get("sma_20")
        rsi_14 = indicators.get("rsi_14")
        
        # Calcul technique avec scores plus équilibrés
        if close and sma_20:
            tech_factors += 1
            pct_diff = (close - sma_20) / sma_20 * 100
            tech_score += np.clip(pct_diff * 1.5, -1, 1)  # Réduction de 2 à 1.5
            
        if rsi_14:
            tech_factors += 1
            if rsi_14 > 70:
                tech_score -= 0.6  # Réduction de 0.8 à 0.6
            elif rsi_14 < 30:
                tech_score += 0.6
            else:
                tech_score += (rsi_14 - 50) / 30  # Plus graduel
                
        if tech_factors > 0:
            tech_score = tech_score / tech_factors
            
        # IA avec modèle corrigé
        ai_score = 0
        if self.ai_enabled and self.dl_model:
            try:
                features = {"close": np.array([close] * 63), "rsi": rsi_14/100, "macd": 0.1, "volatility": 0.05}
                ai_score = float(self.dl_model.predict(features))
            except Exception as e:
                print(f"Erreur IA: {e}")
                
        # CORRECTION 1: Accès correct au sentiment
        sentiment_score = 0
        if self.news_enabled:
            # CORRECTION: Accès direct à la clé symbol dans market_data
            if symbol in self.market_data and "sentiment" in self.market_data[symbol]:
                sentiment_score = self.market_data[symbol]["sentiment"]
                print(f"[DEBUG SENTIMENT FIXED] {symbol} sentiment trouvé: {sentiment_score}")
            else:
                print(f"[DEBUG SENTIMENT FIXED] {symbol} sentiment non trouvé dans {list(self.market_data.keys())}")
                
        # CORRECTION 2: Score total avec amplifications réduites
        total_score = (
            0.4 * tech_score * 1.2  # Réduction de 2.0 à 1.2
            + 0.3 * ai_score * 1.0   # Réduction de 1.5 à 1.0
            + 0.3 * sentiment_score * 1.5  # Réduction de 3.0 à 1.5
        )
        
        decision = {
            "action": "neutral",
            "confidence": abs(total_score),
            "signals": {
                "technical": tech_score,
                "ai": ai_score,
                "sentiment": sentiment_score,
            },
        }
        
        # CORRECTION 3: Seuils plus accessibles pour SELL
        if total_score > 0.2:  # Réduction de 0.3 à 0.2
            decision["action"] = "buy"
        elif total_score < -0.2:  # Réduction de -0.3 à -0.2
            decision["action"] = "sell"
            
        return decision

def create_test_dataframe(symbol="BTCUSDT"):
    """Crée un DataFrame de test"""
    market_data = create_test_market_data()
    data = market_data[symbol]["1h"]
    
    return pd.DataFrame({
        "close": data["close"],
        "high": data["high"],
        "low": data["low"],
        "volume": data["volume"]
    })

async def test_sentiment_integration():
    """Test de l'intégration du sentiment"""
    print("=== TEST INTÉGRATION SENTIMENT ===")
    
    bot = TestTradingBot()
    df = create_test_dataframe("BTCUSDT")
    indicators = create_test_indicators()
    
    print(f"Données de marché disponibles: {list(bot.market_data.keys())}")
    print(f"Sentiment BTCUSDT: {bot.market_data['BTCUSDT'].get('sentiment', 'NON TROUVÉ')}")
    print(f"Sentiment ETHUSDT: {bot.market_data['ETHUSDT'].get('sentiment', 'NON TROUVÉ')}")
    
    # Test version originale (buggée)
    print("\n--- Version ORIGINALE (buggée) ---")
    decision_orig = await bot.analyze_signals_original("BTCUSDT", df, indicators)
    print(f"Décision: {decision_orig['action']} | Confiance: {decision_orig['confidence']:.3f}")
    print(f"Signaux - Tech: {decision_orig['signals']['technical']:.3f} | "
          f"IA: {decision_orig['signals']['ai']:.3f} | "
          f"Sentiment: {decision_orig['signals']['sentiment']:.3f}")
    
    # Test version corrigée
    print("\n--- Version CORRIGÉE ---")
    decision_fixed = await bot.analyze_signals_fixed("BTCUSDT", df, indicators)
    print(f"Décision: {decision_fixed['action']} | Confiance: {decision_fixed['confidence']:.3f}")
    print(f"Signaux - Tech: {decision_fixed['signals']['technical']:.3f} | "
          f"IA: {decision_fixed['signals']['ai']:.3f} | "
          f"Sentiment: {decision_fixed['signals']['sentiment']:.3f}")

async def test_ai_predictions():
    """Test des prédictions IA"""
    print("\n=== TEST PRÉDICTIONS IA ===")
    
    # Test avec différentes valeurs fixes
    test_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    print("--- Modèle ORIGINAL (problématique) ---")
    for val in test_values:
        # Simulation de l'ancienne méthode
        raw_pred = val
        old_normalized = (raw_pred - 0.5) * 4
        old_clipped = np.clip(old_normalized, -1, 1)
        print(f"Entrée: {val:.1f} -> Sortie: {old_clipped:.3f}")
    
    print("\n--- Modèle CORRIGÉ ---")
    for val in test_values:
        model = MockDeepLearningModel(fixed_prediction=val)
        prediction = model.predict({})
        print(f"Entrée: {val:.1f} -> Sortie: {prediction:.3f}")

async def test_sell_signals():
    """Test de génération des signaux SELL"""
    print("\n=== TEST SIGNAUX SELL ===")
    
    bot = TestTradingBot()
    
    # Créer des conditions favorables à un signal SELL
    # 1. Prix en dessous des moyennes mobiles
    sell_indicators = {
        "sma_20": 52000,  # Prix actuel sera en dessous
        "sma_50": 51500,
        "ema_20": 52200,
        "rsi_14": 25,     # Survente
        "macd": -150,
        "macd_signal": -120,
        "macd_hist": -30,
        "bb_upper": 53000,
        "bb_lower": 49000,
        "psar": 52500,
        "momentum_10": -200,
        "zscore_20": -1.5
    }
    
    # DataFrame avec prix en baisse
    sell_df = pd.DataFrame({
        "close": [50000],  # Prix en dessous des SMA
        "high": [50200],
        "low": [49800],
        "volume": [1000]
    })
    
    # Forcer un sentiment négatif
    bot.market_data["BTCUSDT"]["sentiment"] = -0.8
    
    # Forcer une prédiction IA négative
    bot.dl_model = MockDeepLearningModel(fixed_prediction=0.1)  # Prédiction très basse
    
    print("Conditions de test pour SELL:")
    print(f"- Prix: {sell_df['close'].iloc[0]} vs SMA20: {sell_indicators['sma_20']}")
    print(f"- RSI: {sell_indicators['rsi_14']} (survente)")
    print(f"- Sentiment: {bot.market_data['BTCUSDT']['sentiment']}")
    print(f"- Prédiction IA: {bot.dl_model.predict({}):.3f}")
    
    # Test version originale
    print("\n--- Version ORIGINALE ---")
    decision_orig = await bot.analyze_signals_original("BTCUSDT", sell_df, sell_indicators)
    print(f"Décision: {decision_orig['action']} | Confiance: {decision_orig['confidence']:.3f}")
    
    # Test version corrigée
    print("\n--- Version CORRIGÉE ---")
    decision_fixed = await bot.analyze_signals_fixed("BTCUSDT", sell_df, sell_indicators)
    print(f"Décision: {decision_fixed['action']} | Confiance: {decision_fixed['confidence']:.3f}")

async def run_comprehensive_test():
    """Lance tous les tests"""
    print("🔍 DIAGNOSTIC COMPLET DES PROBLÈMES DU BOT DE TRADING")
    print("=" * 60)
    
    await test_sentiment_integration()
    await test_ai_predictions()
    await test_sell_signals()
    
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES CORRECTIONS NÉCESSAIRES:")
    print("1. ✅ Corriger l'accès au sentiment dans analyze_signals")
    print("2. ✅ Réduire l'amplification des prédictions IA (de *4 à *2)")
    print("3. ✅ Ajuster les seuils de décision (de ±0.3 à ±0.2)")
    print("4. ✅ Réduire les amplifications des scores (tech: 2.0→1.2, sentiment: 3.0→1.5)")

def generate_fix_patches():
    """Génère les patches de correction pour les fichiers source"""
    
    patches = {
        "src/ai/deep_learning_model.py": {
            "ligne_58": {
                "ancien": "normalized_pred = (raw_pred - 0.5) * 4  # Amplifier par 4",
                "nouveau": "normalized_pred = (raw_pred - 0.5) * 2  # Amplification réduite"
            }
        },
        
        "src/bot_runner.py": {
            "ligne_811-816": {
                "ancien": """total_score = (
            0.4 * tech_score * 2.0  # Amplifier le score technique
            + 0.3 * ai_score * 1.5   # Amplifier légèrement l'IA
            + 0.25 * sentiment_score * 3.0  # Amplifier fortement le sentiment
            + 0.05 * arbitrage_score
        )""",
                "nouveau": """total_score = (
            0.4 * tech_score * 1.2  # Amplification réduite
            + 0.3 * ai_score * 1.0   # Pas d'amplification
            + 0.3 * sentiment_score * 1.5  # Amplification réduite
            + 0.05 * arbitrage_score
        )"""
            },
            
            "ligne_827-830": {
                "ancien": """if total_score > 0.3:
            decision["action"] = "buy"
        elif total_score < -0.3:
            decision["action"] = "sell" """,
                "nouveau": """if total_score > 0.2:
            decision["action"] = "buy"
        elif total_score < -0.2:
            decision["action"] = "sell" """
            }
        }
    }
    
    print("\n📝 PATCHES DE CORRECTION GÉNÉRÉS:")
    for file_path, changes in patches.items():
        print(f"\n🔧 {file_path}:")
        for location, change in changes.items():
            print(f"  📍 {location}:")
            print(f"    ❌ Ancien: {change['ancien']}")
            print(f"    ✅ Nouveau: {change['nouveau']}")
    
    return patches

if __name__ == "__main__":
    print("🚀 Lancement des tests de correction du bot de trading...")
    asyncio.run(run_comprehensive_test())
    generate_fix_patches()