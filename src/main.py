import os
import sys
import logging
import asyncio
from datetime import datetime
import numpy as np
import ccxt
from dotenv import load_dotenv
import gymnasium as gym
from gymnasium import spaces
import torch
import pandas as pd

# Ajout des chemins pour les modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # Pour accéder à web_interface
sys.path.append(current_dir)  # Pour accéder à src

# Imports des modules existants
from src.data.realtime.websocket.client import MultiStreamManager, StreamConfig
from src.core.buffer.circular_buffer import CircularBuffer
from src.indicators.advanced.multi_timeframe import MultiTimeframeAnalyzer, TimeframeConfig
from src.analysis.indicators.orderflow.orderflow_analysis import OrderFlowAnalysis, OrderFlowConfig
from src.analysis.indicators.volume.volume_analysis import VolumeAnalysis
from src.analysis.indicators.volatility.volatility import VolatilityIndicators
from src.ai.cnn_lstm import CNNLSTM
from src.ai.ppo_gtrxl import PPOGTrXL
from src.ai.hybrid_model import HybridAI
from src.risk_management.circuit_breakers import CircuitBreaker
from src.risk_management.position_manager import PositionManager
from src.core.exchange import ExchangeInterface as Exchange
from src.notifications.telegram_bot import TelegramBot
from src.regime_detection.hmm_kmeans import MarketRegimeDetector
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner as ArbitrageEngine
from src.liquidity_heatmap.visualization import generate_heatmap
from src.monitoring.streamlit_ui import TradingDashboard
from web_interface.app import create_app, socketio
from src.analysis.news.news_integration import NewsAnalyzer

# Configuration
load_dotenv()
config = {
    "TRADING": {
        "base_currency": "USDC",
        "pairs": ["BTC/USDC", "ETH/USDC"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "study_period": "7d"
    },
    "RISK": {
        'max_drawdown': 0.05,
        'daily_stop_loss': 0.02,
        'position_sizing': 'volatility_based',
        'circuit_breaker': {
            'market_crash': True,
            'liquidity_shock': True,
            'black_swan': True
        }
    },
    "AI": {
        "confidence_threshold": 0.75,
        "min_training_size": 1000
    }
}

# Initialisation du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, trading_pairs, timeframes):
        super().__init__()
        self.trading_pairs = trading_pairs
        self.timeframes = timeframes
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(trading_pairs) * len(timeframes) * 42,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(trading_pairs),),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.observation_space.shape)
        return self.state, {}

    def step(self, action):
        reward = 0
        done = False
        truncated = False
        info = {}
        return self.state, reward, done, truncated, info

    def render(self):
        pass

class TradingBotM4:
    def __init__(self):
        """Initialisation du bot"""
        # Initialisation des composants principaux
        self.exchange = Exchange(exchange_id="binance")

        # Configuration et initialisation du stream de données
        stream_config = StreamConfig(
            max_connections=12,
            reconnect_delay=1.0,
            buffer_size=10000
        )
        self.websocket = MultiStreamManager(config=stream_config)
        self.buffer = CircularBuffer()

        # Gestionnaires de trading
        self.position_manager = PositionManager(account_balance=10000)
        self.circuit_breaker = CircuitBreaker()

        # Analyseurs et détecteurs
        self.news_analyzer = NewsAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.arbitrage_engine = ArbitrageEngine()

        # Interface et notifications
        self.telegram = TelegramBot()
        self.dashboard = TradingDashboard(port=8501)
        self.dashboard.current_time = "2025-06-01 05:44:31"
        self.dashboard.current_user = "Patmoorea"

        # Outils de visualisation
        self.generate_heatmap = generate_heatmap        
        self.hybrid_model = HybridAI()
        
        self.env = TradingEnv(
            trading_pairs=config["TRADING"]["pairs"],
            timeframes=config["TRADING"]["timeframes"]
        )
        
        self.decision_model = PPOGTrXL(
            state_dim=256,
            action_dim=len(config["TRADING"]["pairs"]),
            seq_len=64,
            d_model=512,
            num_layers=6,
            num_heads=8
        )
        
        self.timeframe_config = TimeframeConfig(
            timeframes=config["TRADING"]["timeframes"],
            weights={
                "1m": 0.1, "5m": 0.15, "15m": 0.2,
                "1h": 0.25, "4h": 0.15, "1d": 0.15
            }
        )
        self.advanced_indicators = MultiTimeframeAnalyzer(config=self.timeframe_config)
        orderflow_config = OrderFlowConfig(tick_size=0.1)
        self.orderflow_analysis = OrderFlowAnalysis(config=orderflow_config)
        self.volume_analysis = VolumeAnalysis()
        self.volatility_indicators = VolatilityIndicators()

        # Dictionnaire des 42 indicateurs avec leurs méthodes de calcul
        self.indicators = {            "trend": {
                "supertrend": self._calculate_supertrend,
                "ichimoku": self._calculate_ichimoku,
                "vwma": self._calculate_vwma,
                "ema_ribbon": self._calculate_ema_ribbon,
                "parabolic_sar": self._calculate_psar,
                "zigzag": self._calculate_zigzag
            },
            "momentum": {
                "rsi": self._calculate_rsi,
                "stoch_rsi": self._calculate_stoch_rsi,
                "macd": self._calculate_macd,
                "awesome": self._calculate_ao,
                "momentum": self._calculate_momentum,
                "tsi": self._calculate_tsi
            },
            "volatility": {
                "bbands": self._calculate_bbands,
                "keltner": self._calculate_keltner,
                "atr": self._calculate_atr,
                "vix_fix": self._calculate_vix_fix,
                "natr": self._calculate_natr,
                "true_range": self._calculate_tr
            },
            "volume": {
                "obv": self._calculate_obv,
                "vwap": self._calculate_vwap,
                "acc_dist": self._calculate_ad,
                "chaikin_money": self._calculate_cmf,
                "ease_move": self._calculate_emv,
                "volume_profile": self._calculate_vp
            },
            "orderflow": {
                "delta": self._calculate_delta,
                "cvd": self._calculate_cvd,
                "footprint": self._calculate_footprint,
                "liquidity": self._calculate_liquidity,
                "imbalance": self._calculate_imbalance,
                "absorption": self._calculate_absorption
            }
        }
    
    def _calculate_zigzag(self, data, deviation=5.0, backstep=3):
        """Calcule l'indicateur ZigZag"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            
            # Initialisation
            zigzag = pd.Series(index=close.index, dtype=float)
            trend = pd.Series(index=close.index, dtype=int)  # 1 pour haussier, -1 pour baissier
            
            # Trouver les points pivots
            pivot_high = high.rolling(window=2*backstep+1, center=True).max()
            pivot_low = low.rolling(window=2*backstep+1, center=True).min()
            
            # Identifier les points de retournement
            for i in range(backstep, len(close)-backstep):
                if high[i] == pivot_high[i] and high[i] > high[i-1] and high[i] > high[i+1]:
                    trend[i] = 1
                    zigzag[i] = high[i]
                elif low[i] == pivot_low[i] and low[i] < low[i-1] and low[i] < low[i+1]:
                    trend[i] = -1
                    zigzag[i] = low[i]
            
            # Calculer la déviation en pourcentage
            price_change = abs(zigzag.pct_change())
            zigzag[price_change < deviation/100] = np.nan
            
            # Remplir les valeurs manquantes
            zigzag.fillna(method="ffill", inplace=True)
            
            return {
                "zigzag": zigzag,
                "trend": trend,
                "pivots": (pivot_high, pivot_low),
                "strength": price_change * 100
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul ZigZag: {e}")
            return None

    def _calculate_natr(self, data, period=14):
        """Calcule le Normalized Average True Range"""
        try:
            # Calcul de l'ATR
            atr_result = self._calculate_atr(data, period)
            if atr_result is None:
                return None
            
            atr = atr_result["atr"]
            close = data["close"]
            
            # Normalisation de l'ATR
            natr = (atr / close) * 100  # Convertir en pourcentage
            
            return {
                "natr": natr,
                "trend": np.where(natr > natr.shift(1), 1, -1),
                "strength": natr / natr.rolling(window=period).mean()
            }
            
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul NATR: {e}")
            return None

    def _calculate_tr(self, data):
        """Calcule le True Range"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            
            # Calcul des différentes composantes du True Range
            hl = high - low  # Current high-low
            hc = abs(high - close.shift(1))  # High-previous close
            lc = abs(low - close.shift(1))  # Low-previous close
            
            # True Range est le maximum des trois
            tr = pd.DataFrame({
                "hl": hl,
                "hc": hc,
                "lc": lc
            }).max(axis=1)
            
            # Calcul de la variation en pourcentage
            tr_pct = tr / close * 100
            
            return {
                "tr": tr,
                "tr_percent": tr_pct,
                "components": {"hl": hl, "hc": hc, "lc": lc},
                "strength": tr / tr.rolling(window=14).mean()
            }
            
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul True Range: {e}")
            return None

    def _encode_regime(self, regime):
        """Encode le régime de marché en vecteur"""
        regime_mapping = {
            'High Volatility Bull': [1, 0, 0, 0, 0],
            'Low Volatility Bull': [0, 1, 0, 0, 0],
            'High Volatility Bear': [0, 0, 1, 0, 0],
            'Low Volatility Bear': [0, 0, 0, 1, 0],
            'Sideways': [0, 0, 0, 0, 1]
        }
        return regime_mapping.get(regime, [0, 0, 0, 0, 0])

    def _calculate_psar(self, data, af_start=0.02, af_step=0.02, af_max=0.2):
        """Calcule le Parabolic SAR (Stop And Reverse)"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            
            # Initialisation
            psar = pd.Series(index=close.index, dtype=float)
            trend = pd.Series(index=close.index, dtype=int)  # 1 pour haussier, -1 pour baissier
            ep = pd.Series(index=close.index, dtype=float)  # Extreme Point
            af = pd.Series(index=close.index, dtype=float)  # Acceleration Factor
            
            # Valeurs initiales
            trend[0] = 1 if close[0] > close[1] else -1
            psar[0] = low[0] if trend[0] == 1 else high[0]
            ep[0] = high[0] if trend[0] == 1 else low[0]
            af[0] = af_start
            
            # Calcul du PSAR
            for i in range(1, len(close)):
                # Mise à jour du PSAR
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                # Mise à jour du trend
                if trend[i-1] == 1:  # Tendance précédente haussière
                    if low[i] < psar[i]:  # Changement de tendance
                        trend[i] = -1
                        psar[i] = ep[i-1]
                        ep[i] = low[i]
                        af[i] = af_start
                    else:  # Continue tendance haussière
                        trend[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + af_step, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:  # Tendance précédente baissière
                    if high[i] > psar[i]:  # Changement de tendance
                        trend[i] = 1
                        psar[i] = ep[i-1]
                        ep[i] = high[i]
                        af[i] = af_start
                    else:  # Continue tendance baissière
                        trend[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + af_step, af_max)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                
                # Ajustement du PSAR
                if trend[i] == 1:
                    psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                else:
                    psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            
            return {
                "psar": psar,
                "trend": trend,
                "extreme_point": ep,
                "acceleration_factor": af,
                "strength": abs(close - psar) / close
            }
            
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul Parabolic SAR: {e}")
            return None


    def _build_decision(self, policy, value, technical_score, news_sentiment, regime):
        """Construit la décision finale basée sur tous les inputs"""
        # Convertir policy en numpy pour le traitement
        policy_np = policy.detach().numpy()
        
        # Ne garder que les actions d'achat (long only)
        buy_actions = np.maximum(policy_np, 0)
        
        # Calculer la confiance basée sur value et les scores
        confidence = float(np.mean([
            float(value.detach().numpy()),
            technical_score,
            news_sentiment['score']
        ]))
        
        # Trouver le meilleur actif à acheter
        best_pair_idx = np.argmax(buy_actions)
        
        # Construire la décision
        return {
            'action': 'buy' if confidence > config["AI"]["confidence_threshold"] else 'wait',
            'symbol': config["TRADING"]["pairs"][best_pair_idx],
            'entry_price': None,  # Sera défini lors de l'exécution
            'confidence': confidence,
            'regime': regime,
            'technical_score': technical_score,
            'news_impact': news_sentiment['summary'],
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }

    async def initialize(self):  # 4 espaces ici, pas 5
        """Initialisation des composants"""
        logger.info("🚀 Démarrage du Trading Bot M4...")
        await self.websocket.connect()
        await self.telegram.send_message("🤖 Bot démarré et prêt à trader!")

    async def study_market(self, period="7d"):
        """Analyse initiale du marché"""
        logger.info("📊 Étude du marché en cours...")
        try:
            historical_data = await self.exchange.get_historical_data(
                config["TRADING"]["pairs"],
                config["TRADING"]["timeframes"],
                period
            )
            
            indicators_analysis = {}
            for timeframe in config["TRADING"]["timeframes"]:
                tf_data = historical_data[timeframe]
                try:
                    result = self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
                    indicators_analysis[timeframe] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Neutre"
                    } if result is None else result
                except Exception as e:
                    logger.error(f"Erreur analyse {timeframe}: {e}")
                    indicators_analysis[timeframe] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Neutre"
                    }
            
            regime = self.regime_detector.predict(indicators_analysis)
            logger.info(f"📈 Régime de marché détecté: {regime}")
            
            analysis_report = self._generate_analysis_report(indicators_analysis, regime)
            await self.telegram.send_message(analysis_report)
            
            return regime, historical_data, indicators_analysis
            
        except Exception as e:
            logger.error(f"Erreur lors de l'étude du marché: {str(e)}")
            raise
    async def analyze_signals(self, market_data, indicators):  # 4 espaces ici
        """Analyse technique et fondamentale avancée"""
        try:
            technical_score = self.hybrid_model.predict({
                'market_data': market_data,
                'indicators': indicators
            })
            
            news_impact = await self.news_analyzer.analyze_recent_news()
            timeframe_analysis = self._analyze_multi_timeframe(indicators)
            current_regime = self.regime_detector.predict(indicators)
            
            policy, value = self.decision_model.get_action(market_data)
            
            decision = self._build_decision(
                policy=policy,
                value=value,
                technical_score=technical_score,
                news_sentiment=news_impact,
                regime=current_regime
            )
            
            decision = self._add_risk_management(decision)
            return decision
            
        except Exception as e:
            logger.error(f"Erreur analyse signaux: {e}")
            await self.telegram.send_message(f"⚠️ Erreur analyse: {str(e)}")
            return None

    def _generate_analysis_report(self, indicators_analysis, regime, news_sentiment=None):
        """Génère un rapport d'analyse détaillé avec news"""
        current_time = "2025-05-31 19:06:09"  # Mise à jour
        
        report = [
            "📊 Analyse complète du marché:",
            f"Date: {current_time} UTC",
            f"Trader: Patmoorea",
            f"Régime: {regime}",
            "\nTendances principales:"
        ]
        
        # Ajout de l'analyse des news si disponible
        if news_sentiment:
            report.extend([
                "\n📰 Analyse des News:",
                f"Sentiment: {news_sentiment['overall_sentiment']:.2%}",
                f"Impact estimé: {news_sentiment['impact_score']:.2%}",
                f"Événements majeurs: {news_sentiment['major_events']}"
            ])
        
        # Analyse par timeframe
        for timeframe, analysis in indicators_analysis.items():
            report.append(f"\n⏰ {timeframe}:")
            trend_strength = analysis.get('trend', {}).get('trend_strength', 0)
            volatility = analysis.get('volatility', {}).get('current_volatility', 0)
            volume_profile = analysis.get('volume', {}).get('volume_profile', {})
            
            report.extend([
                f"- Force de la tendance: {trend_strength:.2%}",
                f"- Volatilité: {volatility:.2%}",
                f"- Volume: {volume_profile.get('strength', 'N/A')}",
                f"- Signal dominant: {analysis.get('dominant_signal', 'Neutre')}"
            ])
        
        return "\n".join(report)

    async def process_market_data(self):
        """Traitement des données de marché avec tous les indicateurs"""
        try:
            # Récupération des données
            market_data = await self.websocket.get_latest_data()
            
            if not market_data:
                logger.warning("Données de marché manquantes")
                return None, None
            
            # Calcul de tous les indicateurs
            indicators_results = {}
            for timeframe in config["TRADING"]["timeframes"]:
                try:
                    tf_data = market_data.get(timeframe, {})
                    if not tf_data:
                        continue
                    
                    result = self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
                    indicators_results[timeframe] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Neutre"
                    } if result is None else result
                except Exception as e:
                    logger.error(f"Erreur analyse timeframe {timeframe}: {e}")
                    indicators_results[timeframe] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Neutre"
                    }
            
            # Mise à jour du dashboard
            self.dashboard.update(
                market_data,
                indicators_results,
                None,  # heatmap
                current_time="2025-06-01 00:48:28"
            )
            
            return market_data, indicators_results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {e}")
            return None, None            
            # Calcul de tous les indicateurs
            indicators_results = {}
            for timeframe in config["TRADING"]["timeframes"]:
                tf_data = market_data[timeframe]
                indicators_results[timeframe] = self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
            
            # Génération de la heatmap de liquidité
            heatmap = self.generate_heatmap(
                await self.exchange.get_orderbook(config["TRADING"]["pairs"])
            )
            
            # Notification des signaux importants
            await self._notify_significant_signals(indicators_results)
            
            # Mise à jour du dashboard en temps réel
            self.dashboard.update(
                market_data,
                indicators_results,
                heatmap,
                current_time="2025-05-31 05:51:32"
            )
            
            return market_data, indicators_results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {e}")
            await self.telegram.send_message(f"⚠️ Erreur traitement: {str(e)}")
            return None, None

    async def _notify_significant_signals(self, indicators_results):
        """Notifie les signaux importants sur Telegram"""
        current_time = "2025-05-31 05:52:15"  # Mise à jour
        
        for timeframe, analysis in indicators_results.items():
            for category, indicators in analysis.items():
                for indicator, value in indicators.items():
                    if self._is_significant_signal(category, indicator, value):
                        message = (
                            f"⚠️ Signal important détecté!\n"
                            f"Date: {current_time} UTC\n"
                            f"Trader: Patmoorea\n"
                            f"Timeframe: {timeframe}\n"
                            f"Catégorie: {category}\n"
                            f"Indicateur: {indicator}\n"
                            f"Valeur: {value}\n"
                            f"Action suggérée: {'ACHAT' if value > 0 else 'ATTENTE'}"
                        )
                        await self.telegram.send_message(message)

    async def analyze_signals(self, market_data, indicators):
        """Analyse technique et fondamentale avancée"""
        try:
            # Vérification des données
            if market_data is None or indicators is None:
                logger.warning("Données manquantes pour l'analyse")
                return None

            # Utilisation du modèle hybride pour l'analyse technique
            technical_features = self.hybrid_model.analyze_technical(
                market_data=market_data,
                indicators=indicators
            )
            
            # Normalisation si nécessaire
            if not isinstance(technical_features, dict):
                technical_features = {
                    'tensor': technical_features,
                    'score': float(torch.mean(technical_features).item())
                }
            
            # Analyse des news via FinBERT custom
            news_impact = await self.news_analyzer.analyze_recent_news()
            
            # Détection du régime de marché via HMM + K-Means
            current_regime = self.regime_detector.detect_regime(indicators)
            
            # Combinaison des features pour le GTrXL
            combined_features = self._combine_features(
                technical_features,
                news_impact,
                current_regime
            )
            
            # Décision via PPO+GTrXL (6 couches, 512 embeddings)
            policy, value = self.decision_model(combined_features)
            
            # Construction de la décision finale
            decision = self._build_decision(
                policy=policy,
                value=value,
                technical_score=technical_features['score'],
                news_sentiment=news_impact['sentiment'],
                regime=current_regime
            )
            
            # Log de la décision
            logger.info(
                f"Décision générée - Action: {decision['action']}, "
                f"Confiance: {decision['confidence']:.2%}, "
                f"Régime: {decision['regime']}"
            )
            
            # Ajout de la gestion des risques
            decision = self._add_risk_management(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Erreur analyse signaux: {e}")
            await self.telegram.send_message(f"⚠️ Erreur analyse: {str(e)}")
            return None

    def _combine_features(self, technical_features, news_impact, regime):
        """Combine toutes les features pour le GTrXL"""
        try:
            # Conversion en tensors
            technical_tensor = technical_features['tensor']
            news_tensor = torch.tensor(news_impact['embeddings'], dtype=torch.float32)
            regime_tensor = torch.tensor(self._encode_regime(regime), dtype=torch.float32)
            
            # Ajout de dimensions si nécessaire
            if news_tensor.dim() == 1:
                news_tensor = news_tensor.unsqueeze(0)
            if regime_tensor.dim() == 1:
                regime_tensor = regime_tensor.unsqueeze(0)
            
            # Combinaison
            features = torch.cat([
                technical_tensor,
                news_tensor,
                regime_tensor
            ], dim=-1)
            
            return features
            
        except Exception as e:
            logger.error(f"Erreur lors de la combinaison des features: {e}")
            raise

    async def execute_trades(self, decision):
        """Exécution des trades selon la décision"""
        current_time = "2025-05-31 05:52:53"  # Mise à jour
        
        # Vérification du circuit breaker
        if await self.circuit_breaker.should_stop_trading():
            logger.warning("🛑 Circuit breaker activé - Trading suspendu")
            await self.telegram.send_message(
                "⚠️ Trading suspendu: Circuit breaker activé\n"
                f"Date: {current_time} UTC\n"
                f"Trader: Patmoorea"
            )
            return
        
        if decision and decision["confidence"] > config["AI"]["confidence_threshold"]:
            try:
                # Vérification des opportunités d'arbitrage
                arb_ops = await self.arbitrage_engine.find_opportunities()
                if arb_ops:
                    await self.telegram.send_message(
                        f"💰 Opportunité d'arbitrage détectée:\n"
                        f"Date: {current_time} UTC\n"
                        f"Trader: Patmoorea\n"
                        f"Details: {arb_ops}"
                    )
                
                # Récupération du prix actuel
                current_price = await self.exchange.get_price(decision["symbol"])
                decision["entry_price"] = current_price
                
                # Calcul de la taille de position avec gestion du risque
                position_size = self.position_manager.calculate_position_size(
                    decision,
                    available_balance=await self.exchange.get_balance("USDC")
                )
                
                # Vérification finale avant l'ordre
                if not self._validate_trade(decision, position_size):
                    logger.warning("Trade invalidé par les vérifications finales")
                    return
                
                # Placement de l'ordre avec stop loss
                order = await self.exchange.create_order(
                    symbol=decision["symbol"],
                    type="limit",
                    side="buy",  # Achat uniquement comme demandé
                    amount=position_size,
                    price=decision["entry_price"],
                    params={
                        "stopLoss": {
                            "type": "trailing",
                            "activation_price": decision["trailing_stop"]["activation_price"],
                            "callback_rate": decision["trailing_stop"]["callback_rate"]
                        },
                        "takeProfit": {
                            "price": decision["take_profit"]
                        }
                    }
                )
                
                # Notification Telegram détaillée
                await self.telegram.send_message(
                    f"🔄 Ordre placé:\n"
                    f"Date: {current_time} UTC\n"
                    f"Trader: Patmoorea\n"
                    f"Symbol: {order['symbol']}\n"
                    f"Type: {order['type']}\n"
                    f"Prix: {order['price']}\n"
                    f"Stop Loss: {decision['stop_loss']}\n"
                    f"Take Profit: {decision['take_profit']}\n"
                    f"Trailing Stop: {decision['trailing_stop']['activation_price']}\n"
                    f"Confiance: {decision['confidence']:.2%}\n"
                    f"Régime: {decision['regime']}\n"
                    f"News Impact: {decision['news_impact']}\n"
                    f"Volume: {position_size} {config['TRADING']['base_currency']}"
                )
                
                # Mise à jour du dashboard
                self.dashboard.update_trades(order)
                
            except Exception as e:
                logger.error(f"Erreur lors de l'exécution: {e}")
                await self.telegram.send_message(
                    f"⚠️ Erreur d'exécution: {str(e)}\n"
                    f"Date: {current_time} UTC\n"
                    f"Trader: Patmoorea"
                )

    def _validate_trade(self, decision, position_size):
        """Validation finale avant l'exécution du trade"""
        try:
            # Vérification de la taille minimale
            if position_size < 0.001:  # Exemple de taille minimale
                logger.warning("Taille de position trop petite")
                return False
            
            # Vérification du spread
            if self._check_spread_too_high(decision["symbol"]):
                logger.warning("Spread trop important")
                return False
            
            # Vérification de la liquidité
            if not self._check_sufficient_liquidity(decision["symbol"], position_size):
                logger.warning("Liquidité insuffisante")
                return False
            
            # Vérification des news à haut risque
            if self._check_high_risk_news():
                logger.warning("News à haut risque détectées")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation du trade: {e}")
            return False

    async def run(self):
        """Boucle principale du bot"""
        current_time = "2025-05-31 05:53:37"  # Mise à jour
        current_user = "Patmoorea"
        
        try:
            await self.initialize()
            
            # Banner de démarrage
            logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║                 Trading Bot Ultimate v4 Started               ║
╠══════════════════════════════════════════════════════════════╣
║ Time: {current_time} UTC                                     ║
║ User: {current_user}                                         ║
║ Mode: BUY_ONLY                                              ║
║ AI: PPO-GTrXL (6-layer, 512d)                              ║
║ Status: RUNNING                                             ║
╚══════════════════════════════════════════════════════════════╝
            """)
            
            # Étude initiale du marché
            regime, historical_data, initial_analysis = await self.study_market(
                config["TRADING"]["study_period"]
            )
            
            # Entraînement initial si nécessaire
            if self._should_train(historical_data):
                await self._train_models(historical_data, initial_analysis)
            
            while True:
                try:
                    # 1. Traitement des données
                    market_data, indicators = await self.process_market_data()
                    if market_data is None or indicators is None:
                        logger.warning("Données manquantes, attente...")
                        await asyncio.sleep(5)
                        continue
                    
                    # 2. Analyse et décision
                    decision = await self.analyze_signals(market_data, indicators)
                    
                    # 3. Mise à jour du régime de marché si nécessaire
                    current_regime = self.regime_detector.detect_regime(indicators)
                    if current_regime != regime:
                        regime = current_regime
                        logger.info(f"Changement de régime détecté: {regime}")
                        await self.telegram.send_message(
                            f"📊 Changement de régime détecté!\n"
                            f"Date: {current_time} UTC\n"
                            f"Nouveau régime: {regime}"
                        )
                    
                    # 4. Exécution si nécessaire
                    if decision and decision.get('action') == 'buy':
                        await self.execute_trades(decision)
                    
                    # 5. Mise à jour du dashboard
                    self.dashboard.update_status({
                        'time': current_time,
                        'user': current_user,
                        'regime': regime,
                        'last_decision': decision
                    })
                    
                    # Attente avant la prochaine itération
                    await asyncio.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("👋 Arrêt manuel demandé")
                    await self.telegram.send_message(
                        f"🛑 Bot arrêté manuellement\n"
                        f"Date: {current_time} UTC\n"
                        f"User: {current_user}"
                    )
                    break
                    
                except Exception as e:
                    logger.error(f"Erreur critique: {e}")
                    await self.telegram.send_message(
                        f"🚨 Erreur critique: {str(e)}\n"
                        f"Date: {current_time} UTC\n"
                        f"User: {current_user}"
                    )
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
            await self.telegram.send_message(
                f"💀 Erreur fatale - Bot arrêté: {str(e)}\n"
                f"Date: {current_time} UTC\n"
                f"User: {current_user}"
            )
            raise

    def _should_train(self, historical_data):
        """Détermine si les modèles doivent être réentraînés"""
        return len(historical_data.get('1h', [])) >= config["AI"]["min_training_size"]

    async def _train_models(self, historical_data, initial_analysis):
        """Entraîne ou met à jour les modèles"""
        try:
            logger.info("🧠 Début de l'entraînement des modèles...")
            
            # Entraînement du modèle hybride
            self.hybrid_model.train(
                market_data=historical_data,
                indicators=initial_analysis
            )
            
            # Entraînement du PPO-GTrXL
            self.decision_model.train_step((
                historical_data,
                initial_analysis,
                self._calculate_advantages(historical_data),
                self._calculate_returns(historical_data),
                self._get_old_policies(historical_data)
            ))
            
            logger.info("✅ Entraînement terminé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {e}")
            raise

    # Méthodes de calcul des indicateurs de tendance
    def _calculate_supertrend(self, data, period=10, multiplier=3):
        """Calcule l'indicateur Supertrend"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calcul de l'ATR
            atr = self.volatility_indicators.calculate_atr(data, period)
            
            # Calcul des bandes
            upperband = ((high + low) / 2) + (multiplier * atr)
            lowerband = ((high + low) / 2) - (multiplier * atr)
            
            # Calcul du Supertrend
            supertrend = pd.Series(index=close.index)
            direction = pd.Series(index=close.index)
            
            for i in range(period, len(close)):
                if close[i] > upperband[i-1]:
                    direction[i] = 1
                elif close[i] < lowerband[i-1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i-1]
                    
                if direction[i] == 1:
                    supertrend[i] = lowerband[i]
                else:
                    supertrend[i] = upperband[i]
                    
            return {
                'value': supertrend,
                'direction': direction,
                'strength': abs(close - supertrend) / close
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul Supertrend: {e}")
            return None

    def _calculate_ichimoku(self, data, params=(9, 26, 52)):
        """Calcule l'indicateur Ichimoku"""
        try:
            high = data['high']
            low = data['low']
            
            tenkan_period, kijun_period, senkou_period = params
            
            # Tenkan-sen (Conversion Line)
            tenkan = (high.rolling(window=tenkan_period).max() +
                     low.rolling(window=tenkan_period).min()) / 2
            
            # Kijun-sen (Base Line)
            kijun = (high.rolling(window=kijun_period).max() +
                    low.rolling(window=kijun_period).min()) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
            
            # Senkou Span B (Leading Span B)
            senkou_b = ((high.rolling(window=senkou_period).max() +
                        low.rolling(window=senkou_period).min()) / 2).shift(kijun_period)
            
            # Chikou Span (Lagging Span)
            chikou = data['close'].shift(-kijun_period)
            
            return {
                'tenkan': tenkan,
                'kijun': kijun,
                'senkou_a': senkou_a,
                'senkou_b': senkou_b,
                'chikou': chikou,
                'cloud_strength': abs(senkou_a - senkou_b) / data['close']
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul Ichimoku: {e}")
            return None

    def _calculate_vwma(self, data, period=20):
        """Calcule la Volume Weighted Moving Average"""
        try:
            vwma = (data['close'] * data['volume']).rolling(window=period).sum() / \
                   data['volume'].rolling(window=period).sum()
            
            return {
                'value': vwma,
                'trend': np.where(vwma > vwma.shift(1), 1, -1),
                'strength': abs(data['close'] - vwma) / data['close']
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul VWMA: {e}")
            return None

    def _calculate_ema_ribbon(self, data, periods=[5,10,20,50,100,200]):
        """Calcule le EMA Ribbon"""
        try:
            emas = {}
            for period in periods:
                emas[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
            
            # Calcul de la force de la tendance basée sur l'alignement des EMAs
            trend_strength = 0
            for i in range(len(periods)-1):
                if emas[f'ema_{periods[i]}'].iloc[-1] > emas[f'ema_{periods[i+1]}'].iloc[-1]:
                    trend_strength += 1
                else:
                    trend_strength -= 1
                    
            return {
                'emas': emas,
                'trend': np.sign(trend_strength),
                'strength': abs(trend_strength) / (len(periods) - 1)
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul EMA Ribbon: {e}")
            return None

    # Méthodes de calcul des indicateurs de momentum
    def _calculate_rsi(self, data, period=14):
        """Calcule le Relative Strength Index"""
        try:
            close = data['close']
            delta = close.diff()
            
            # Séparation des gains et pertes
            gain = (delta.where(delta > 0, 0)).ewm(span=period).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return {
                'value': rsi,
                'overbought': rsi > 70,
                'oversold': rsi < 30,
                'divergence': self._check_divergence(close, rsi)
            }
            
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul RSI: {e}")
            return None

    def _calculate_stoch_rsi(self, data, period=14, k_period=3, d_period=3):
        """Calcule le Stochastic RSI"""
        try:
            # Calcul du RSI
            rsi = self._calculate_rsi(data, period)['value']
            
            # Calcul du Stochastic RSI
            stoch_rsi = (rsi - rsi.rolling(period).min()) / \
                       (rsi.rolling(period).max() - rsi.rolling(period).min())
            
            # Lignes K et D
            k_line = stoch_rsi.rolling(k_period).mean() * 100
            d_line = k_line.rolling(d_period).mean()
            
            return {
                'k_line': k_line,
                'd_line': d_line,
                'overbought': k_line > 80,
                'oversold': k_line < 20,
                'crossover': self._detect_crossover(k_line, d_line)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:06] Erreur calcul Stoch RSI: {e}")
            return None

    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calcule le MACD (Moving Average Convergence Divergence)"""
        try:
            close = data['close']
            
            # Calcul des EMA
            fast_ema = close.ewm(span=fast, adjust=False).mean()
            slow_ema = close.ewm(span=slow, adjust=False).mean()
            
            # Calcul du MACD et de sa ligne de signal
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram,
                'crossover': self._detect_crossover(macd_line, signal_line),
                'strength': abs(histogram) / close
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:06] Erreur calcul MACD: {e}")
            return None

    def _calculate_ao(self, data, fast=5, slow=34):
        """Calcule l'Awesome Oscillator"""
        try:
            # Calcul des médianes
            median_price = (data['high'] + data['low']) / 2
            
            # Calcul des SMA
            fast_sma = median_price.rolling(window=fast).mean()
            slow_sma = median_price.rolling(window=slow).mean()
            
            # Calcul de l'AO
            ao = fast_sma - slow_sma
            
            # Détection des changements de momentum
            momentum_shift = np.sign(ao.diff())
            
            return {
                'value': ao,
                'momentum_shift': momentum_shift,
                'strength': abs(ao) / median_price,
                'zero_cross': self._detect_zero_cross(ao)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:06] Erreur calcul Awesome Oscillator: {e}")
            return None

    def _calculate_momentum(self, data, period=10):
        """Calcule l'indicateur de Momentum"""
        try:
            close = data['close']
            momentum = close / close.shift(period) * 100
            
            return {
                'value': momentum,
                'trend': np.where(momentum > 100, 1, -1),
                'strength': abs(momentum - 100) / 100,
                'acceleration': momentum.diff()
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:06] Erreur calcul Momentum: {e}")
            return None

    def _calculate_tsi(self, data, slow=25, fast=13, signal=13):
        """Calcule le True Strength Index"""
        try:
            close = data['close']
            diff = close.diff()
            
            # Double smoothing des prix
            smooth1 = diff.ewm(span=slow).mean()
            smooth2 = smooth1.ewm(span=fast).mean()
            
            # Double smoothing de la valeur absolue
            abs_diff = abs(diff)
            abs_smooth1 = abs_diff.ewm(span=slow).mean()
            abs_smooth2 = abs_smooth1.ewm(span=fast).mean()
            
            # Calcul du TSI
            tsi = (smooth2 / abs_smooth2) * 100
            signal_line = tsi.ewm(span=signal).mean()
            
            return {
                'tsi': tsi,
                'signal': signal_line,
                'histogram': tsi - signal_line,
                'divergence': self._check_divergence(close, tsi)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:06] Erreur calcul TSI: {e}")
            return None

    # Méthodes de calcul des indicateurs de volatilité
    def _calculate_bbands(self, data, period=20, std_dev=2):
        """Calcule les Bandes de Bollinger"""
        try:
            close = data['close']
            
            # Calcul de la moyenne mobile
            middle_band = close.rolling(window=period).mean()
            
            # Calcul de l'écart-type
            std = close.rolling(window=period).std()
            
            # Calcul des bandes
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            # Calcul de la largeur des bandes
            bandwidth = (upper_band - lower_band) / middle_band
            
            # Calcul du %B
            percent_b = (close - lower_band) / (upper_band - lower_band)
            
            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band,
                'bandwidth': bandwidth,
                'percent_b': percent_b,
                'squeeze': bandwidth < bandwidth.rolling(window=period).mean()
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:48] Erreur calcul Bollinger Bands: {e}")
            return None

    def _calculate_keltner(self, data, period=20, atr_mult=2):
        """Calcule les Bandes de Keltner"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Calcul de l'EMA
            middle_line = close.ewm(span=period).mean()
            
            # Calcul de l'ATR
            atr = self.volatility_indicators.calculate_atr(data, period)
            
            # Calcul des bandes
            upper_band = middle_line + (atr_mult * atr)
            lower_band = middle_line - (atr_mult * atr)
            
            return {
                'upper': upper_band,
                'middle': middle_line,
                'lower': lower_band,
                'width': (upper_band - lower_band) / middle_line,
                'position': (close - lower_band) / (upper_band - lower_band)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:48] Erreur calcul Keltner Channels: {e}")
            return None

    def _calculate_atr(self, data, period=14):
        """Calcule l'Average True Range"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calcul du True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # Calcul de l'ATR
            atr = true_range.ewm(span=period).mean()
            
            # Calcul de la volatilité normalisée
            normalized_atr = atr / close
            
            return {
                'value': atr,
                'normalized': normalized_atr,
                'trend': atr.diff().apply(np.sign),
                'volatility_regime': pd.qcut(normalized_atr, q=3, labels=['Low', 'Medium', 'High'])
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:48] Erreur calcul ATR: {e}")
            return None

    def _calculate_vix_fix(self, data, period=22):
        """Calcule le VIX Fix (Volatility Index Fix)"""
        try:
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Calcul des log returns
            log_returns = np.log(close / close.shift(1))
            
            # Calcul de la volatilité historique
            hist_vol = log_returns.rolling(window=period).std() * np.sqrt(252)
            
            # Calcul du VIX Fix
            price_range = (high - low) / close
            vix_fix = hist_vol * price_range.rolling(window=period).mean() * 100
            
            return {
                'value': vix_fix,
                'regime': pd.qcut(vix_fix, q=3, labels=['Low', 'Medium', 'High']),
                'trend': vix_fix.diff().apply(np.sign),
                'percentile': vix_fix.rank(pct=True)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:55:48] Erreur calcul VIX Fix: {e}")
            return None

    # Méthodes de calcul des indicateurs de volume
    def _calculate_vwap(self, data):
        """Calcule le Volume Weighted Average Price"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']
            
            # Calcul du prix typique
            typical_price = (high + low + close) / 3
            
            # Calcul du VWAP
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            
            # Calcul des bandes de déviation
            std = (typical_price - vwap) * volume
            std = (std ** 2).cumsum() / volume.cumsum()
            std = np.sqrt(std)
            
            upper_band = vwap + (2 * std)
            lower_band = vwap - (2 * std)
            
            return {
                'vwap': vwap,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'deviation': (close - vwap) / vwap,
                'volume_trend': self._calculate_volume_trend(volume)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:56:30] Erreur calcul VWAP: {e}")
            return None

    def _calculate_obv(self, data):
        """Calcule l'On Balance Volume"""
        try:
            close = data['close']
            volume = data['volume']
            
            # Calcul de l'OBV
            close_diff = close.diff()
            obv = volume.copy()
            obv[close_diff < 0] = -volume[close_diff < 0]
            obv[close_diff == 0] = 0
            obv = obv.cumsum()
            
            # Calcul de l'EMA de l'OBV
            obv_ema = obv.ewm(span=20).mean()
            
            return {
                'value': obv,
                'ema': obv_ema,
                'trend': np.where(obv > obv.shift(1), 1, -1),
                'divergence': self._check_divergence(close, obv),
                'strength': abs(obv - obv_ema) / obv_ema
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:56:30] Erreur calcul OBV: {e}")
            return None

    def _calculate_volume_profile(self, data, price_levels=100):
        """Calcule le Volume Profile"""
        try:
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            # Création des niveaux de prix
            price_range = np.linspace(low.min(), high.max(), price_levels)
            volume_profile = pd.Series(index=price_range, data=0.0)
            
            # Distribution du volume sur les niveaux de prix
            for i in range(len(data)):
                level_volume = volume.iloc[i] / price_levels
                price_levels_in_range = (price_range >= low.iloc[i]) & (price_range <= high.iloc[i])
                volume_profile[price_levels_in_range] += level_volume
            
            # Calcul du Point of Control (POC)
            poc_price = price_range[volume_profile.argmax()]
            
            # Calcul des Value Area
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.70  # 70% du volume
            
            sorted_profile = volume_profile.sort_values(ascending=False)
            value_area_indices = sorted_profile.cumsum() <= value_area_volume
            value_area = sorted_profile[value_area_indices]
            
            return {
                'profile': volume_profile,
                'poc': poc_price,
                'value_area_high': value_area.index.max(),
                'value_area_low': value_area.index.min(),
                'distribution_shape': self._analyze_distribution_shape(volume_profile)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:56:30] Erreur calcul Volume Profile: {e}")
            return None

    # Méthodes de calcul des indicateurs d'orderflow
    def _calculate_delta(self, data):
        """Calcule le Delta (différence entre volume d'achat et de vente)"""
        try:
            # Récupération des données d'orderflow
            buy_volume = data['buy_volume']
            sell_volume = data['sell_volume']
            
            # Calcul du delta
            delta = buy_volume - sell_volume
            cumulative_delta = delta.cumsum()
            
            # Calcul des divergences
            price_trend = data['close'].diff().apply(np.sign)
            delta_trend = delta.diff().apply(np.sign)
            divergence = price_trend != delta_trend
            
            return {
                'delta': delta,
                'cumulative_delta': cumulative_delta,
                'buy_pressure': buy_volume / (buy_volume + sell_volume),
                'divergence': divergence,
                'strength': abs(delta) / (buy_volume + sell_volume)
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:56:30] Erreur calcul Delta: {e}")
            return None

    def _calculate_liquidity(self, data, levels=10):
        """Analyse la liquidité dans le carnet d'ordres"""
        try:
            # Récupération des données du carnet d'ordres
            bids = data['bids']  # [[price, volume], ...]
            asks = data['asks']  # [[price, volume], ...]
            
            # Calcul de la liquidité cumulée
            bid_liquidity = np.cumsum([vol for _, vol in bids[:levels]])
            ask_liquidity = np.cumsum([vol for _, vol in asks[:levels]])
            
            # Détection des niveaux de liquidité importants
            bid_clusters = self._detect_liquidity_clusters(bids)
            ask_clusters = self._detect_liquidity_clusters(asks)
            
            return {
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'imbalance': (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity),
                'bid_clusters': bid_clusters,
                'ask_clusters': ask_clusters,
                'spread': asks[0][0] - bids[0][0]
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:56:30] Erreur calcul Liquidity: {e}")
            return None

    def _detect_liquidity_clusters(self, orders, threshold=0.1):
        """Détecte les clusters de liquidité dans le carnet d'ordres"""
        try:
            clusters = []
            total_volume = sum(vol for _, vol in orders)
            current_cluster = {'start_price': orders[0][0], 'volume': 0}
            
            for price, volume in orders:
                volume_ratio = volume / total_volume
                
                if volume_ratio >= threshold:
                    if current_cluster['volume'] == 0:
                        current_cluster['start_price'] = price
                    current_cluster['volume'] += volume
                else:
                    if current_cluster['volume'] > 0:
                        current_cluster['end_price'] = price
                        clusters.append(current_cluster)
                        current_cluster = {'start_price': price, 'volume': 0}
                        
            return clusters
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:57:16] Erreur détection clusters: {e}")
            return []

    def _check_divergence(self, price, indicator, lookback=10):
        """Détecte les divergences entre le prix et un indicateur"""
        try:
            price_high = price.rolling(window=lookback).max()
            price_low = price.rolling(window=lookback).min()
            ind_high = indicator.rolling(window=lookback).max()
            ind_low = indicator.rolling(window=lookback).min()
            
            bullish_div = (price_low < price_low.shift(1)) & (ind_low > ind_low.shift(1))
            bearish_div = (price_high > price_high.shift(1)) & (ind_high < ind_high.shift(1))
            
            return pd.Series(index=price.index, data=np.where(bullish_div, 1, np.where(bearish_div, -1, 0)))
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:57:16] Erreur détection divergence: {e}")
            return pd.Series(0, index=price.index)

    def _add_risk_management(self, decision):
        """Ajoute les paramètres de gestion des risques à la décision"""
        try:
            if decision['action'] != 'buy':
                return decision
                
            # Calcul de l'ATR pour le stop loss
            atr = self.volatility_indicators.calculate_atr(
                self.buffer.get_latest()[decision['timeframe']]
            )['value'].iloc[-1]
            
            # Configuration du stop loss
            stop_multiplier = 2.0  # Ajustable selon le régime de marché
            stop_distance = atr * stop_multiplier
            
            decision['stop_loss'] = decision['entry_price'] - stop_distance
            
            # Configuration du take profit
            reward_ratio = 2.0  # Risk:Reward ratio minimum
            decision['take_profit'] = decision['entry_price'] + (stop_distance * reward_ratio)
            
            # Configuration du trailing stop
            decision['trailing_stop'] = {
                'activation_price': decision['entry_price'] + (stop_distance * 1.5),
                'callback_rate': 0.5  # 50% de la distance ATR
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:57:16] Erreur ajout risk management: {e}")
            return decision

    def _calculate_ad(self, data):
        """Calcule l'indicateur Accumulation/Distribution"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            volume = data["volume"]
            
            # Calcul du Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            
            # Calcul du Money Flow Volume
            mfv = mfm * volume
            
            # Calcul de l'AD
            ad = mfv.cumsum()
            
            return {
                "ad": ad,
                "trend": np.where(ad > ad.shift(1), 1, -1),
                "strength": abs(ad - ad.rolling(window=14).mean()) / ad,
                "divergence": self._check_divergence(close, ad)
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul AD: {e}")
            return None

    def _calculate_cmf(self, data, period=20):
        """Calcule le Chaikin Money Flow"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            volume = data["volume"]
            
            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)
            
            # Money Flow Volume
            mfv = mfm * volume
            
            # Chaikin Money Flow
            cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            
            return {
                "value": cmf,
                "trend": np.where(cmf > 0, 1, -1),
                "strength": abs(cmf),
                "divergence": self._check_divergence(close, cmf)
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul CMF: {e}")
            return None

    def _calculate_emv(self, data, volume_divisor=10000):
        """Calcule l'Ease of Movement"""
        try:
            high = data["high"]
            low = data["low"]
            volume = data["volume"]
            
            # Distance Moved
            distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
            
            # Box Ratio
            box_ratio = (volume / volume_divisor) / (high - low)
            
            # Ease of Movement
            emv = distance / box_ratio
            emv_ma = emv.rolling(window=14).mean()
            
            return {
                "value": emv,
                "ma": emv_ma,
                "trend": np.where(emv > emv_ma, 1, -1),
                "strength": abs(emv - emv_ma) / emv_ma
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul EMV: {e}")
            return None

    def _calculate_cvd(self, data):
        """Calcule le Cumulative Volume Delta"""
        try:
            # Récupération des données d'orderflow
            buy_volume = data["buy_volume"]
            sell_volume = data["sell_volume"]
            close = data["close"]
            
            # Calcul du delta
            delta = buy_volume - sell_volume
            
            # Calcul du CVD
            cvd = delta.cumsum()
            cvd_ma = cvd.rolling(window=20).mean()
            
            return {
                "value": cvd,
                "ma": cvd_ma,
                "trend": np.where(cvd > cvd_ma, 1, -1),
                "strength": abs(cvd - cvd_ma) / cvd_ma,
                "divergence": self._check_divergence(close, cvd)
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul CVD: {e}")
            return None

    def _calculate_footprint(self, data):
        """Calcule le Footprint Chart"""
        try:
            # Récupération des données d'orderflow par niveau de prix
            price_levels = data["price_levels"]
            buy_volume = data["buy_volume_by_price"]
            sell_volume = data["sell_volume_by_price"]
            
            # Calcul du delta par niveau de prix
            delta_by_price = buy_volume - sell_volume
            
            # Identification des zones d'accumulation/distribution
            accumulation = delta_by_price > 0
            distribution = delta_by_price < 0
            
            return {
                "delta_by_price": delta_by_price,
                "accumulation_zones": price_levels[accumulation],
                "distribution_zones": price_levels[distribution],
                "volume_profile": buy_volume + sell_volume,
                "imbalance_ratio": abs(delta_by_price) / (buy_volume + sell_volume)
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul Footprint: {e}")
            return None

    def _calculate_imbalance(self, data):
        """Calcule les déséquilibres de marché"""
        try:
            bids = data["bids"]  # [[price, volume], ...]
            asks = data["asks"]  # [[price, volume], ...]
            
            # Calcul des déséquilibres
            bid_volume = sum(vol for _, vol in bids)
            ask_volume = sum(vol for _, vol in asks)
            
            # Ratio de déséquilibre
            imbalance_ratio = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Détection des fair value gaps
            gaps = self._detect_fair_value_gaps(data)
            
            return {
                "ratio": imbalance_ratio,
                "bid_strength": bid_volume / (bid_volume + ask_volume),
                "ask_strength": ask_volume / (bid_volume + ask_volume),
                "gaps": gaps,
                "status": "buying_pressure" if imbalance_ratio > 0.2 else "selling_pressure" if imbalance_ratio < -0.2 else "balanced"
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul Imbalance: {e}")
            return None

    def _calculate_absorption(self, data):
        """Calcule l'absorption du marché"""
        try:
            # Données d'orderflow
            trades = data["trades"]  # [[price, volume, side], ...]
            
            # Calcul de l'absorption
            buy_trades = [t for t in trades if t[2] == "buy"]
            sell_trades = [t for t in trades if t[2] == "sell"]
            
            # Ratio d'absorption
            buy_volume = sum(t[1] for t in buy_trades)
            sell_volume = sum(t[1] for t in sell_trades)
            
            absorption_ratio = buy_volume / sell_volume if sell_volume > 0 else float("inf")
            
            return {
                "ratio": absorption_ratio,
                "buy_pressure": buy_volume / (buy_volume + sell_volume),
                "sell_pressure": sell_volume / (buy_volume + sell_volume),
                "efficiency": min(buy_volume, sell_volume) / max(buy_volume, sell_volume),
                "status": "absorbing" if absorption_ratio > 1.5 else "distributing" if absorption_ratio < 0.67 else "neutral"
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul Absorption: {e}")
            return None

    def _detect_fair_value_gaps(self, data):
        """Détecte les fair value gaps dans le carnet d'ordres"""
        try:
            candles = data["candles"]  # [[time, open, high, low, close], ...]
            gaps = []
            
            for i in range(1, len(candles)):
                prev_close = candles[i-1][4]
                curr_open = candles[i][1]
                
                if curr_open > prev_close * 1.001:  # Gap haussier de 0.1%
                    gaps.append({
                        "type": "bullish",
                        "start": prev_close,
                        "end": curr_open
                    })
                elif curr_open < prev_close * 0.999:  # Gap baissier de 0.1%
                    gaps.append({
                        "type": "bearish",
                        "start": prev_close,
                        "end": curr_open
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur détection gaps: {e}")
            return []

    def _calculate_vp(self, data, price_levels=100):
        """Calcule le Volume Profile"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            volume = data["volume"]
            
            # Création des niveaux de prix
            price_range = np.linspace(low.min(), high.max(), price_levels)
            volume_profile = pd.Series(index=price_range, data=0.0)
            
            # Distribution du volume sur les niveaux de prix
            for i in range(len(data)):
                level_volume = volume.iloc[i] / price_levels
                price_levels_in_range = (price_range >= low.iloc[i]) & (price_range <= high.iloc[i])
                volume_profile[price_levels_in_range] += level_volume
            
            # Calcul du Point of Control (POC)
            poc_price = price_range[volume_profile.argmax()]
            
            # Value Area (70% du volume)
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.70
            sorted_profile = volume_profile.sort_values(ascending=False)
            value_area = sorted_profile[sorted_profile.cumsum() <= value_area_volume]
            
            # Analyse de la distribution
            distribution_shape = self._analyze_distribution_shape(volume_profile)
            
            return {
                "profile": volume_profile,
                "poc": poc_price,
                "value_area_high": value_area.index.max(),
                "value_area_low": value_area.index.min(),
                "distribution": distribution_shape,
                "volume_concentration": value_area.sum() / total_volume
            }
            
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul Volume Profile: {e}")
            return None


    def _calculate_vp(self, data, price_levels=100):
        """Calcule le Volume Profile"""
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            volume = data["volume"]
            
            # Création des niveaux de prix
            price_range = np.linspace(low.min(), high.max(), price_levels)
            volume_profile = pd.Series(index=price_range, data=0.0)
            
            # Distribution du volume sur les niveaux de prix
            for i in range(len(data)):
                level_volume = volume.iloc[i] / price_levels
                price_levels_in_range = (price_range >= low.iloc[i]) & (price_range <= high.iloc[i])
                volume_profile[price_levels_in_range] += level_volume
            
            # Calcul du Point of Control (POC)
            poc_price = price_range[volume_profile.argmax()]
            
            # Value Area (70% du volume)
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.70
            sorted_profile = volume_profile.sort_values(ascending=False)
            value_area = sorted_profile[sorted_profile.cumsum() <= value_area_volume]
            
            # Analyse de la distribution
            distribution_shape = self._analyze_distribution_shape(volume_profile)
            
            return {
                "profile": volume_profile,
                "poc": poc_price,
                "value_area_high": value_area.index.max(),
                "value_area_low": value_area.index.min(),
                "distribution": distribution_shape,
                "volume_concentration": value_area.sum() / total_volume
            }
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur calcul Volume Profile: {e}")
            return None

    def _analyze_distribution_shape(self, profile):
        """Analyse la forme de la distribution du volume"""
        try:
            # Calcul des statistiques de base
            mean = np.average(profile.index, weights=profile.values)
            std = np.sqrt(np.average((profile.index - mean)**2, weights=profile.values))
            
            # Calcul de la skewness et kurtosis
            skewness = np.average(((profile.index - mean)/std)**3, weights=profile.values)
            kurtosis = np.average(((profile.index - mean)/std)**4, weights=profile.values)
            
            # Classification de la distribution
            if abs(skewness) < 0.5 and abs(kurtosis - 3) < 1:
                shape = "Normal"
            elif kurtosis > 4:
                shape = "Leptokurtic"  # Pics prononcés
            else:
                shape = "Non-normal"
                
            return {
                'shape': shape,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'std': std
            }
            
        except Exception as e:
            logger.error(f"[2025-05-31 05:57:16] Erreur analyse distribution: {e}")
            return None

# Fonction principale
async def main():
    """Point d'entrée principal du bot"""
    try:
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        
        # Banner de démarrage
        print(f"""
╔════════════════════════════════════════════════════════════╗
║             Trading Bot Ultimate v4 - Starting             ║
╠════════════════════════════════════════════════════════════╣
║ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC      ║
║ User: Patmoorea                                            ║
║ Mode: Production                                           ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        # Création et démarrage du bot
        bot = TradingBotM4()
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Arrêt manuel du bot")
        
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        raise
        
    finally:
        # Nettoyage final
        logging.shutdown()

if __name__ == "__main__":
    # Exécution du bot
    asyncio.run(main())

    def _analyze_trend_signals(self, trend_data):
        """Analyse les signaux de tendance"""
        if not trend_data:
            return None
            
        signals = []
        
        # Analyse du Supertrend
        if 'supertrend' in trend_data:
            st = trend_data['supertrend']
            if st['signal'] == 1 and st['strength'] > 0.7:
                signals.append(f"Supertrend Haussier (Force: {st['strength']:.2%})")
            elif st['signal'] == -1 and st['strength'] > 0.7:
                signals.append(f"Supertrend Baissier (Force: {st['strength']:.2%})")

        # Analyse de l'EMA Ribbon
        if 'ema_ribbon' in trend_data:
            ribbon = trend_data['ema_ribbon']
            if ribbon['bullish_alignment'] and ribbon['strength'] > 0.8:
                signals.append(f"EMA Ribbon Haussier (Force: {ribbon['strength']:.2%})")
            elif ribbon['bearish_alignment'] and ribbon['strength'] > 0.8:
                signals.append(f"EMA Ribbon Baissier (Force: {ribbon['strength']:.2%})")

        return ' | '.join(signals) if signals else None

    def _analyze_volatility_signals(self, volatility_data):
        """Analyse les signaux de volatilité"""
        if not volatility_data:
            return None
            
        signals = []
        
        # Analyse des Bandes de Bollinger
        if 'bbands' in volatility_data:
            bb = volatility_data['bbands']
            if bb['bandwidth'] > bb['bandwidth_high']:
                signals.append(f"Forte Volatilité BB (BW: {bb['bandwidth']:.2f})")
            elif bb['bandwidth'] < bb['bandwidth_low']:
                signals.append(f"Faible Volatilité BB (BW: {bb['bandwidth']:.2f})")

        # Analyse de l'ATR
        if 'atr' in volatility_data:
            atr = volatility_data['atr']
            if atr['value'] > atr['high_level']:
                signals.append(f"ATR Élevé ({atr['value']:.2f})")

        return ' | '.join(signals) if signals else None

    def _analyze_volume_signals(self, volume_data):
        """Analyse les signaux de volume"""
        if not volume_data:
            return None
            
        signals = []
        
        # Analyse du Volume Profile
        if 'volume_profile' in volume_data:
            vp = volume_data['volume_profile']
            if vp['poc_strength'] > 0.8:
                signals.append(f"POC Fort ({vp['poc_price']:.2f})")

        # Analyse du VWAP
        if 'vwap' in volume_data:
            vwap = volume_data['vwap']
            if vwap['deviation'] > 2.0:
                signals.append(f"Déviation VWAP ({vwap['deviation']:.2f}σ)")

        return ' | '.join(signals) if signals else None

    async def _update_dashboard(self, market_data, indicators, heatmap, current_time):
        """Met à jour le dashboard avec les dernières données"""
        try:
            self.dashboard.update(
                market_data=market_data,
                indicators=indicators,
                heatmap=heatmap,
                current_time=current_time,
                additional_info={
                    "performance": self._calculate_performance(),
                    "active_signals": self._get_active_signals(),
                    "risk_metrics": self._calculate_risk_metrics()
                }
            )
        except Exception as e:
            logger.error(f"Erreur mise à jour dashboard: {e}")

