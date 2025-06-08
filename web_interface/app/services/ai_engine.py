import tensorflow as tf
import numpy as np
from datetime import datetime, timezone
import logging
from typing import Dict, List, Tuple

class CNNLSTMModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = self._build_model()

    def _build_model(self):
        try:
            # Input shape: (batch_size, timesteps, features)
            input_shape = (None, 100, 5)
            
            model = tf.keras.Sequential([
                # Input Layer
                tf.keras.layers.Input(shape=input_shape[1:]),
                
                # 1D CNN layers instead of 2D
                tf.keras.layers.Conv1D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Conv1D(128, 3, activation='relu'),
                tf.keras.layers.MaxPooling1D(2),
                
                # LSTM layers
                tf.keras.layers.LSTM(256, return_sequences=True),
                tf.keras.layers.LSTM(128),
                
                # Dense layers
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                
                # Output layer
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building CNN-LSTM model: {e}")
            return None

class AIDecisionEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cnn_lstm = CNNLSTMModel()
        self.last_analysis = None
        self.market_conditions = {
            'trend': 'neutral',
            'volatility': 'normal',
            'volume': 'normal',
            'sentiment': 'neutral'
        }

    async def analyze_market(self, market_data: Dict) -> Dict:
        try:
            # Préparation des données
            features = self._prepare_features(market_data)
            
            # Prédiction du modèle (si disponible)
            model_prediction = 0.5
            if self.cnn_lstm.model is not None:
                try:
                    model_prediction = float(self.cnn_lstm.model.predict(features, verbose=0)[0][0])
                except Exception as e:
                    self.logger.warning(f"Model prediction failed: {e}")
            
            # Analyse technique
            technical_score = self._analyze_technical_indicators(market_data)
            
            # Analyse du marché
            market_score = self._analyze_market_conditions(market_data)
            
            # Calcul de la confiance globale
            confidence = (model_prediction + technical_score + market_score) / 3
            
            # Décision
            action = self._make_decision(confidence, market_data)
            
            analysis_result = {
                "action": action,
                "confidence": float(confidence),
                "model_prediction": float(model_prediction),
                "technical_score": float(technical_score),
                "market_score": float(market_score),
                "market_conditions": self.market_conditions,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.last_analysis = analysis_result
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "model_prediction": 0.0,
                "technical_score": 0.0,
                "market_score": 0.0,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _prepare_features(self, market_data: Dict) -> np.ndarray:
        try:
            # Préparer un tableau de features de forme (1, 100, 5)
            features = np.zeros((1, 100, 5))
            
            if 'technical' in market_data:
                data = market_data['technical']
                # Remplir les features avec les données disponibles
                if 'close' in data:
                    features[0, :, 0] = self._normalize(data['close'][-100:])
                if 'volume' in data:
                    features[0, :, 1] = self._normalize(data['volume'][-100:])
                if 'rsi' in data:
                    features[0, :, 2] = self._normalize(data['rsi'][-100:])
                if 'macd' in data:
                    features[0, :, 3] = self._normalize(data['macd'][-100:])
                if 'bbands' in data:
                    features[0, :, 4] = self._normalize(data['bbands'][-100:])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return np.zeros((1, 100, 5))

    def _normalize(self, data: List[float]) -> np.ndarray:
        if not data:
            return np.zeros(100)
        data = np.array(data[-100:])  # Prendre les 100 dernières valeurs
        if len(data) < 100:
            # Padding si nécessaire
            padding = np.zeros(100 - len(data))
            data = np.concatenate([padding, data])
        return (data - np.mean(data)) / (np.std(data) + 1e-8)

    def _analyze_technical_indicators(self, market_data: Dict) -> float:
        try:
            scores = []
            
            if 'technical' in market_data:
                if 'rsi' in market_data['technical']:
                    scores.append(self._evaluate_rsi(market_data['technical']['rsi']))
                if 'macd' in market_data['technical']:
                    scores.append(self._evaluate_macd(market_data['technical']['macd']))
                if 'bbands' in market_data['technical']:
                    scores.append(self._evaluate_bbands(market_data['technical']))
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical indicators: {e}")
            return 0.5

    def _analyze_market_conditions(self, market_data: Dict) -> float:
        try:
            # Analyse des conditions du marché
            trend = self._detect_trend(market_data)
            volatility = self._calculate_volatility(market_data)
            volume = self._analyze_volume(market_data)
            
            # Mise à jour des conditions du marché
            self.market_conditions.update({
                'trend': trend,
                'volatility': volatility,
                'volume': volume
            })
            
            # Calcul du score
            scores = {
                'trend': {'bullish': 1.0, 'neutral': 0.5, 'bearish': 0.0},
                'volatility': {'high': 0.3, 'normal': 1.0, 'low': 0.7},
                'volume': {'high': 1.0, 'normal': 0.7, 'low': 0.3}
            }
            
            return np.mean([
                scores['trend'][trend],
                scores['volatility'][volatility],
                scores['volume'][volume]
            ])
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return 0.5

    def _make_decision(self, confidence: float, market_data: Dict) -> str:
        try:
            # Mode achat uniquement
            if confidence > 0.75 and self.market_conditions['trend'] == 'bullish':
                return "BUY"
            else:
                return "HOLD"
                
        except Exception as e:
            self.logger.error(f"Error making decision: {e}")
            return "HOLD"

    def _detect_trend(self, market_data: Dict) -> str:
        try:
            if 'technical' in market_data and 'close' in market_data['technical']:
                prices = market_data['technical']['close'][-20:]  # Derniers 20 points
                if len(prices) > 1:
                    trend = np.mean(np.diff(prices))
                    if trend > 0.001:
                        return 'bullish'
                    elif trend < -0.001:
                        return 'bearish'
            return 'neutral'
        except Exception as e:
            self.logger.error(f"Error detecting trend: {e}")
            return 'neutral'

    def _calculate_volatility(self, market_data: Dict) -> str:
        try:
            if 'technical' in market_data and 'close' in market_data['technical']:
                prices = market_data['technical']['close'][-20:]
                if len(prices) > 1:
                    volatility = np.std(np.diff(prices) / prices[:-1])
                    if volatility > 0.02:
                        return 'high'
                    elif volatility < 0.005:
                        return 'low'
            return 'normal'
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 'normal'

    def _analyze_volume(self, market_data: Dict) -> str:
        try:
            if 'technical' in market_data and 'volume' in market_data['technical']:
                volumes = market_data['technical']['volume'][-20:]
                avg_volume = np.mean(volumes)
                current_volume = volumes[-1]
                
                if current_volume > avg_volume * 1.5:
                    return 'high'
                elif current_volume < avg_volume * 0.5:
                    return 'low'
            return 'normal'
        except Exception as e:
            self.logger.error(f"Error analyzing volume: {e}")
            return 'normal'
