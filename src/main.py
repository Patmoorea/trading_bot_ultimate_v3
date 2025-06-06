"""
Trading Bot Ultimate v4
Version: 4.0.0
Last Updated: 2025-06-06 01:20:02 UTC
Author: Patmoorea  
Status: PRODUCTION

Features:
- Multi-timeframe analysis 
- Advanced risk management
- AI-powered decision making
- Real-time market regime detection
"""
import streamlit as st
st.set_page_config(
    page_title="Trading Bot Ultimate v4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
import os
os.environ['STREAMLIT_HIDE_PYTORCH_WARNING'] = '1'  # Supprime les warnings Torch
import sys
import logging
import asyncio
import nest_asyncio
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
sys.path.append(parent_dir)
sys.path.append(current_dir)

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
        "min_training_size": 1000,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "n_epochs": 10,
        "gtrxl_layers": 6,
        "embedding_dim": 512,
        "dropout": 0.1,
        "gradient_clip": 0.5
    },
    "INDICATORS": {
        "trend": {
            "supertrend": {
                "period": 10,
                "multiplier": 3
            },
            "ichimoku": {
                "tenkan": 9,
                "kijun": 26,
                "senkou": 52
            },
            "ema_ribbon": [5, 10, 20, 50, 100, 200]
        },
        "momentum": {
            "rsi": {
                "period": 14,
                "overbought": 70,
                "oversold": 30
            },
            "stoch_rsi": {
                "period": 14,
                "k": 3,
                "d": 3
            },
            "macd": {
                "fast": 12,
                "slow": 26,
                "signal": 9
            }
        },
        "volatility": {
            "bbands": {
                "period": 20,
                "std_dev": 2
            },
            "keltner": {
                "period": 20,
                "atr_mult": 2
            },
            "atr": {
                "period": 14
            }
        },
        "volume": {
            "vwap": {
                "anchor": "session"
            },
            "obv": {
                "signal": 20
            },
            "volume_profile": {
                "price_levels": 100
            }
        },
        "orderflow": {
            "delta": {
                "window": 100
            },
            "cvd": {
                "smoothing": 20
            },
            "imbalance": {
                "threshold": 0.2
            }
        }
    }
}

# Initialisation du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """Environment d'apprentissage par renforcement pour le trading"""
    
    def __init__(self, trading_pairs, timeframes):
        super().__init__()
        self.trading_pairs = trading_pairs
        self.timeframes = timeframes
        
        # Espace d'observation: 42 features par paire/timeframe
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(len(trading_pairs) * len(timeframes) * 42,),
            dtype=np.float32
        )
        
        # Espace d'action: allocation par paire entre 0 et 1
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(trading_pairs),),
            dtype=np.float32
        )
        
        # Paramètres d'apprentissage
        self.reward_scale = 1.0
        self.position_history = []
        self.done_penalty = -1.0
        
        # Initialisation des métriques
        self.metrics = {
            'episode_rewards': [],
            'portfolio_values': [],
            'positions': [],
            'actions': []
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.observation_space.shape)
        self.position_history.clear()
        return self.state, {}
        
    def step(self, action):
        # Validation de l'action
        if not self.action_space.contains(action):
            logger.warning(f"Action invalide: {action}")
            action = np.clip(action, self.action_space.low, self.action_space.high)
            
        # Calcul de la récompense
        reward = self._calculate_reward(action)
        
        # Mise à jour de l'état
        self._update_state()
        
        # Vérification des conditions de fin
        done = self._check_done()
        truncated = False
        
        # Mise à jour des métriques
        self._update_metrics(action, reward)
        
        return self.state, reward, done, truncated, self._get_info()

    def _calculate_reward(self, action):
        """Calcule la récompense basée sur le PnL et le risque"""
        try:
            # Calcul du PnL
            pnl = self._calculate_pnl(action)
            
            # Pénalité pour le risque
            risk_penalty = self._calculate_risk_penalty(action)
            
            # Reward final
            reward = (pnl - risk_penalty) * self.reward_scale
            
            return float(reward)
            
        except Exception as e:
            logger.error(f"Erreur calcul reward: {e}")
            return 0.0

    def _update_state(self):
        """Mise à jour de l'état avec les dernières données de marché"""
        try:
            # Mise à jour des features techniques
            technical_features = self._calculate_technical_features()
            
            # Mise à jour des features de marché
            market_features = self._calculate_market_features()
            
            # Combinaison des features
            self.state = np.concatenate([technical_features, market_features])
            
        except Exception as e:
            logger.error(f"Erreur mise à jour state: {e}")

    def _check_done(self):
        """Vérifie les conditions de fin d'épisode"""
        # Vérification du stop loss
        if self._check_stop_loss():
            return True
            
        # Vérification de la durée max
        if len(self.position_history) >= self.max_steps:
            return True
            
        return False

    def _update_metrics(self, action, reward):
        """Mise à jour des métriques de l'épisode"""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['portfolio_values'].append(self._get_portfolio_value())
        self.metrics['positions'].append(self.position_history[-1])
        self.metrics['actions'].append(action)

    def _get_info(self):
        """Retourne les informations additionnelles"""
        return {
            'portfolio_value': self._get_portfolio_value(),
            'current_positions': self.position_history[-1] if self.position_history else None,
            'metrics': self.metrics
        }

    def render(self):
        """Affichage de l'environnement"""
        # Affichage des métriques principales
        print(f"\nPortfolio Value: {self._get_portfolio_value():.2f}")
        print(f"Total Reward: {sum(self.metrics['episode_rewards']):.2f}")
        print(f"Number of Trades: {len(self.position_history)}")

class MultiStreamManager:
    def __init__(self, pairs=None, config=None):
        """Initialise le gestionnaire de flux multiples"""
        self.pairs = pairs or []
        self.config = config
        self.exchange = None  # Initialisé plus tard
        self.buffer = CircularBuffer()
        
    def setup_exchange(self, exchange_id="binance"):
        """Configure l'exchange"""
        self.exchange = Exchange(exchange_id=exchange_id)

class TradingBotM4:
    """Classe principale du bot de trading v4"""
    def __init__(self):
         # Récupération des variables d'environnement
        self.current_user = os.getenv('CURRENT_USER', 'Patmoorea')
        self.current_time = os.getenv('CURRENT_TIME', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        self.trading_mode = os.getenv('TRADING_MODE', 'production')
        
        # Configuration de l'exchange et des streams
        self.stream_config = StreamConfig(
            max_connections=12,
            reconnect_delay=1.0,
            buffer_size=10000
        )
        
        # Initialisation du MultiStreamManager
        self.websocket = MultiStreamManager(
            pairs=config["TRADING"]["pairs"],
            config=self.stream_config
        )
        
        # Configuration de l'exchange
        self.websocket.setup_exchange("binance")
        
        self.buffer = CircularBuffer()
        self.buffer = CircularBuffer()

        # Interface et monitoring
        self.dashboard = TradingDashboard()
        self.current_time = "2025-06-06 01:20:02"
        self.current_user = "Patmoorea"

        # Composants principaux
        self.arbitrage_engine = ArbitrageEngine()
        self.telegram = TelegramBot()

        # IA et analyse
        self.hybrid_model = HybridAI()
        self.env = TradingEnv(
            trading_pairs=config["TRADING"]["pairs"],
            timeframes=config["TRADING"]["timeframes"]
        )
        
        # Gestionnaires de trading 
        self.position_manager = PositionManager(
            account_balance=10000,
            max_positions=5,
            max_leverage=3.0,
            min_position_size=0.001
        )
        self.circuit_breaker = CircuitBreaker(
            crash_threshold=0.1,
            liquidity_threshold=0.5,
            volatility_threshold=0.3
        )
        # Configuration des timeframes et indicateurs
        self.timeframe_config = TimeframeConfig(
            timeframes=config["TRADING"]["timeframes"],
            weights={
                "1m": 0.1, "5m": 0.15, "15m": 0.2,
                "1h": 0.25, "4h": 0.15, "1d": 0.15
            }
        )
        
        # Initialisation des analyseurs
        self._initialize_analyzers()
        
    def _initialize_analyzers(self):
        """Initialize all analysis components"""
        self.advanced_indicators = MultiTimeframeAnalyzer(
            config=self.timeframe_config
        )
        self.orderflow_analysis = OrderFlowAnalysis(
            config=OrderFlowConfig(tick_size=0.1)
        )
        self.volume_analysis = VolumeAnalysis()
        self.volatility_indicators = VolatilityIndicators()
        
        # Dictionnaire des 42 indicateurs avec leurs méthodes de calcul
        self.indicators = {
            "trend": {
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

        # Initialisation des modèles d'IA
        self.models = {
            "cnn_lstm": CNNLSTM(
                input_size=42,
                hidden_size=256,
                num_layers=3,
                dropout=config["AI"]["dropout"]
            ),
            "ppo_gtrxl": PPOGTrXL(
                state_dim=42 * len(config["TRADING"]["timeframes"]),
                action_dim=len(config["TRADING"]["pairs"]),
                n_layers=config["AI"]["gtrxl_layers"],
                embedding_dim=config["AI"]["embedding_dim"]
            )
        }

    async def get_latest_data(self):
        """Récupère les dernières données de marché"""
        try:
            data = {}
            for pair in config["TRADING"]["pairs"]:
                for timeframe in config["TRADING"]["timeframes"]:
                    data.setdefault(timeframe, {})[pair] = self.buffer.get_latest()
            return data
        except Exception as e:
            logger.error(f"[{datetime.utcnow()}] Erreur get_latest_data: {e}")
            return None

    async def study_market(self, period="7d"):
        """Analyse initiale du marché"""
        logger.info("🔊 Étude du marché en cours...")
        current_time = "2025-06-06 07:39:24"  # Mise à jour timestamp
        
        try:
            # Récupération des données historiques
            historical_data = await self.exchange.get_historical_data(
                config["TRADING"]["pairs"],
                config["TRADING"]["timeframes"],
                period
            )
            
            # Analyse des indicateurs par timeframe
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
                    logger.error(f"[{current_time}] Erreur analyse {timeframe}: {e}")
                    
            # Détection du régime de marché
            regime = self.regime_detector.predict(indicators_analysis)
            logger.info(f"🔈 Régime de marché détecté: {regime}")
            
            # Génération et envoi du rapport
            analysis_report = self._generate_analysis_report(
                indicators_analysis, 
                regime,
                current_time=current_time
            )
            await self.telegram.send_message(analysis_report)
            
            # Mise à jour du dashboard
            self.dashboard.update_market_analysis(
                historical_data=historical_data,
                indicators=indicators_analysis,
                regime=regime,
                timestamp=current_time
            )
            
            return regime, historical_data, indicators_analysis
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur lors de l'étude du marché: {str(e)}")
            raise

    async def analyze_signals(self, market_data, indicators):
        """Analyse technique et fondamentale avancée"""
        current_time = "2025-06-06 07:39:24"  # Mise à jour timestamp
        
        try:
            # Vérification des données
            if market_data is None or indicators is None:
                logger.warning(f"[{current_time}] Données manquantes pour l'analyse")
                return None

            # Utilisation du modèle hybride pour l'analyse technique
            technical_features = self.hybrid_model.analyze_technical(
                market_data=market_data,
                indicators=indicators,
                timestamp=current_time
            )
            
            # Normalisation si nécessaire
            if not isinstance(technical_features, dict):
                technical_features = {
                    'tensor': technical_features,
                    'score': float(torch.mean(technical_features).item())
                }
            
            # Analyse des news via FinBERT custom
            news_impact = await self.news_analyzer.analyze_recent_news(
                timestamp=current_time
            )
            
            # Détection du régime de marché via HMM + K-Means
            current_regime = self.regime_detector.detect_regime(
                indicators,
                timestamp=current_time
            )
            
            # Combinaison des features pour le GTrXL
            combined_features = self._combine_features(
                technical_features,
                news_impact,
                current_regime
            )
            
            # Décision via PPO+GTrXL (6 couches, 512 embeddings)
            policy, value = self.decision_model(
                combined_features,
                timestamp=current_time
            )
            
            # Construction de la décision finale
            decision = self._build_decision(
                policy=policy,
                value=value,
                technical_score=technical_features['score'],
                news_sentiment=news_impact['sentiment'],
                regime=current_regime,
                timestamp=current_time
            )
            
            # Ajout de la gestion des risques
            decision = self._add_risk_management(
                decision,
                timestamp=current_time
            )
            
            # Log de la décision
            logger.info(
                f"[{current_time}] Décision générée - "
                f"Action: {decision['action']}, "
                f"Confiance: {decision['confidence']:.2%}, "
                f"Régime: {decision['regime']}"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur analyse signaux: {e}")
            await self.telegram.send_message(
                f"⚠️ Erreur analyse: {str(e)}\n"
                f"Date: {current_time} UTC\n"
                f"Trader: {self.current_user}"
            )
            return None

    def _build_decision(self, policy, value, technical_score, news_sentiment, regime, timestamp):
        """Construit la décision finale basée sur tous les inputs"""
        try:
            # Conversion policy en numpy pour le traitement
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
            decision = {
                "action": "buy" if confidence > config["AI"]["confidence_threshold"] else "wait",
                "symbol": config["TRADING"]["pairs"][best_pair_idx],
                "confidence": confidence,
                "timestamp": timestamp,
                "regime": regime,
                "technical_score": technical_score,
                "news_impact": news_sentiment['sentiment'],
                "value_estimate": float(value.detach().numpy()),
                "position_size": buy_actions[best_pair_idx]
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"[{timestamp}] Erreur construction décision: {e}")
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
            logger.error(f"[2025-06-06 07:40:42] Erreur lors de la combinaison des features: {e}")
            raise

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

    async def execute_trades(self, decision):
        """Exécution des trades selon la décision"""
        current_time = "2025-06-06 07:40:42"  # Mise à jour timestamp
        
        # Vérification du circuit breaker
        if await self.circuit_breaker.should_stop_trading():
            logger.warning(f"[{current_time}] 🛑 Circuit breaker activé - Trading suspendu")
            await self.telegram.send_message(
                "⚠️ Trading suspendu: Circuit breaker activé\n"
                f"Date: {current_time} UTC\n"
                f"Trader: {self.current_user}"
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
                        f"Trader: {self.current_user}\n"
                        f"Details: {arb_ops}"
                    )
                
                # Récupération du prix actuel
                current_price = await self.exchange.get_price(decision["symbol"])
                decision["entry_price"] = current_price
                
                # Calcul de la taille de position avec gestion du risque
                position_size = self.position_manager.calculate_position_size(
                    decision,
                    available_balance=await self.exchange.get_balance(config["TRADING"]["base_currency"])
                )
                
                # Vérification finale avant l'ordre
                if not self._validate_trade(decision, position_size):
                    logger.warning(f"[{current_time}] Trade invalidé par les vérifications finales")
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
                    f"📄 Ordre placé:\n"
                    f"Date: {current_time} UTC\n"
                    f"Trader: {self.current_user}\n"
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
                logger.error(f"[{current_time}] Erreur lors de l'exécution: {e}")
                await self.telegram.send_message(
                    f"⚠️ Erreur d'exécution: {str(e)}\n"
                    f"Date: {current_time} UTC\n"
                    f"Trader: {self.current_user}"
                )

    def _validate_trade(self, decision, position_size):
        """Validation finale avant l'exécution du trade"""
        current_time = "2025-06-06 07:40:42"  # Mise à jour timestamp
        
        try:
            # Vérification de la taille minimale
            if position_size < 0.001:  # Exemple de taille minimale
                logger.warning(f"[{current_time}] Taille de position trop petite")
                return False
            
            # Vérification du spread
            if self._check_spread_too_high(decision["symbol"]):
                logger.warning(f"[{current_time}] Spread trop important")
                return False
            
            # Vérification de la liquidité
            if not self._check_sufficient_liquidity(decision["symbol"], position_size):
                logger.warning(f"[{current_time}] Liquidité insuffisante")
                return False
            
            # Vérification des news à haut risque
            if self._check_high_risk_news():
                logger.warning(f"[{current_time}] News à haut risque détectées")
                return False
            
            # Vérification des limites de position
            if not self.position_manager.check_position_limits(position_size):
                logger.warning(f"[{current_time}] Limites de position dépassées")
                return False
            
            # Vérification du timing d'entrée
            if not self._check_entry_timing(decision):
                logger.warning(f"[{current_time}] Timing d'entrée non optimal")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur lors de la validation du trade: {e}")
            return False

    def _check_spread_too_high(self, symbol):
        """Vérifie si le spread est trop important"""
        try:
            orderbook = self.buffer.get_orderbook(symbol)
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            
            spread = (best_ask - best_bid) / best_bid
            return spread > 0.001  # 0.1% spread maximum
            
        except Exception as e:
            logger.error(f"[2025-06-06 07:40:42] Erreur vérification spread: {e}")
            return True  # Par sécurité

    def _check_sufficient_liquidity(self, symbol, position_size):
        """Vérifie s'il y a assez de liquidité pour le trade"""
        try:
            orderbook = self.buffer.get_orderbook(symbol)
            
            # Calcul de la profondeur de marché nécessaire
            required_liquidity = position_size * 3  # 3x la taille pour la sécurité
            
            # Somme de la liquidité disponible
            available_liquidity = sum(vol for _, vol in orderbook['bids'][:10])
            
            return available_liquidity >= required_liquidity
            
        except Exception as e:
            logger.error(f"[2025-06-06 07:40:42] Erreur vérification liquidité: {e}")
            return False

    def _check_entry_timing(self, decision):
        """Vérifie si le timing d'entrée est optimal"""
        try:
            # Vérification des signaux de momentum
            momentum_signals = self._analyze_momentum_signals()
            if momentum_signals["strength"] < 0.5:
                return False
                
            # Vérification de la volatilité
            volatility = self._analyze_volatility()
            if volatility["current"] > volatility["threshold"]:
                return False
                
            # Vérification du volume
            volume_analysis = self._analyze_volume_profile()
            if not volume_analysis["supports_entry"]:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"[2025-06-06 07:40:42] Erreur vérification timing: {e}")
            return False
    def _analyze_momentum_signals(self):
        """Analyse des signaux de momentum"""
        current_time = "2025-06-06 07:41:38"  # Mise à jour timestamp
        
        try:
            signals = {
                "rsi": self._calculate_rsi(self.buffer.get_latest()),
                "macd": self._calculate_macd(self.buffer.get_latest()),
                "stoch": self._calculate_stoch_rsi(self.buffer.get_latest())
            }
            
            # Calcul de la force globale
            strengths = []
            if signals["rsi"]:
                strengths.append(abs(signals["rsi"]["strength"]))
            if signals["macd"]:
                strengths.append(abs(signals["macd"]["strength"]))
            if signals["stoch"]:
                strengths.append(abs(signals["stoch"]["strength"]))
                
            return {
                "signals": signals,
                "strength": np.mean(strengths) if strengths else 0,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur analyse momentum: {e}")
            return {"strength": 0, "timestamp": current_time}

    def _analyze_volatility(self):
        """Analyse de la volatilité actuelle"""
        current_time = "2025-06-06 07:41:38"  # Mise à jour timestamp
        
        try:
            # Calcul des indicateurs de volatilité
            bbands = self._calculate_bbands(self.buffer.get_latest())
            atr = self._calculate_atr(self.buffer.get_latest())
            
            # Calcul de la volatilité normalisée
            current_volatility = 0
            if bbands and atr:
                bb_width = bbands["bandwidth"]
                atr_norm = atr["normalized"]
                current_volatility = (bb_width + atr_norm) / 2
                
            return {
                "current": current_volatility,
                "threshold": 0.8,  # Seuil dynamique basé sur le régime
                "timestamp": current_time,
                "indicators": {
                    "bbands": bbands,
                    "atr": atr
                }
            }
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur analyse volatilité: {e}")
            return {"current": 1, "threshold": 0, "timestamp": current_time}

    def _analyze_volume_profile(self):
        """Analyse du profil de volume"""
        current_time = "2025-06-06 07:41:38"  # Mise à jour timestamp
        
        try:
            vp = self._calculate_vp(self.buffer.get_latest())
            
            if not vp:
                return {"supports_entry": False, "timestamp": current_time}
                
            # Analyse des niveaux de support/résistance
            current_price = self.buffer.get_latest()["close"].iloc[-1]
            nearest_poc = min(vp["poc"], key=lambda x: abs(x - current_price))
            
            # Vérification des conditions d'entrée
            price_near_poc = abs(current_price - nearest_poc) / current_price < 0.01
            volume_increasing = vp["profile"][-1] > np.mean(vp["profile"][-5:])
            
            return {
                "supports_entry": price_near_poc and volume_increasing,
                "poc": nearest_poc,
                "volume_trend": "increasing" if volume_increasing else "decreasing",
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur analyse volume profile: {e}")
            return {"supports_entry": False, "timestamp": current_time}

    async def run(self):
        """Boucle principale du bot"""
        current_time = "2025-06-06 07:41:38"  # Mise à jour timestamp
        current_user = self.current_user
        
        try:
            # Banner de démarrage
            logger.info(f"""
╔═════════════════════════════════════════════════════════════╗
║                Trading Bot Ultimate v4 Started               ║
╠═════════════════════════════════════════════════════════════╣
║ Time: {current_time} UTC                                    ║
║ User: {current_user}                                        ║
║ Mode: BUY_ONLY                                             ║
║ AI: PPO-GTrXL (6-layer, 512d)                             ║
║ Status: RUNNING                                            ║
╚═════════════════════════════════════════════════════════════╝
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
                        logger.warning(f"[{current_time}] Données manquantes, attente...")
                        await asyncio.sleep(5)
                        continue
                    
                    # 2. Analyse et décision
                    decision = await self.analyze_signals(market_data, indicators)
                    
                    # 3. Mise à jour du régime de marché si nécessaire
                    current_regime = self.regime_detector.detect_regime(indicators)
                    if current_regime != regime:
                        regime = current_regime
                        logger.info(f"[{current_time}] Changement de régime détecté: {regime}")
                        await self.telegram.send_message(
                            f"🔈 Changement de régime détecté!\n"
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
                        'last_decision': decision,
                        'performance_metrics': self._calculate_performance_metrics()
                    })
                    
                    # 6. Vérification des conditions d'arrêt
                    if await self._should_stop_trading():
                        logger.info(f"[{current_time}] Conditions d'arrêt atteintes")
                        break
                    
                    # Attente avant la prochaine itération
                    await asyncio.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info(f"[{current_time}] ❌ Arrêt manuel demandé")
                    await self.telegram.send_message(
                        f"🛑 Bot arrêté manuellement\n"
                        f"Date: {current_time} UTC\n"
                        f"User: {current_user}"
                    )
                    break
                    
                except Exception as e:
                    logger.error(f"[{current_time}] Erreur critique: {e}")
                    await self.telegram.send_message(
                        f"🚨 Erreur critique: {str(e)}\n"
                        f"Date: {current_time} UTC\n"
                        f"User: {current_user}"
                    )
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"[{current_time}] Erreur fatale: {e}")
            await self.telegram.send_message(
                f"💀 Erreur fatale - Bot arrêté: {str(e)}\n"
                f"Date: {current_time} UTC\n"
                f"User: {current_user}"
            )
            raise

    def _should_train(self, historical_data):
        """Détermine si les modèles doivent être réentraînés"""
        try:
            # Vérification de la taille minimale des données
            if len(historical_data.get('1h', [])) < config["AI"]["min_training_size"]:
                return False
                
            # Vérification de la dernière session d'entraînement
            if not hasattr(self, 'last_training_time'):
                return True
                
            time_since_training = datetime.utcnow() - self.last_training_time
            return time_since_training.days >= 1  # Réentraînement quotidien
            
        except Exception as e:
            logger.error(f"[2025-06-06 07:41:38] Erreur vérification entraînement: {e}")
            return False
    async def _train_models(self, historical_data, initial_analysis):
        """Entraîne ou met à jour les modèles"""
        current_time = "2025-06-06 07:42:32"  # Mise à jour timestamp
        
        try:
            logger.info(f"[{current_time}] 🎮 Début de l'entraînement des modèles...")
            
            # Préparation des données d'entraînement
            X_train, y_train = self._prepare_training_data(
                historical_data,
                initial_analysis
            )
            
            # Entraînement du modèle hybride
            self.hybrid_model.train(
                market_data=historical_data,
                indicators=initial_analysis,
                epochs=config["AI"]["n_epochs"],
                batch_size=config["AI"]["batch_size"],
                learning_rate=config["AI"]["learning_rate"]
            )
            
            # Entraînement du PPO-GTrXL
            self.models["ppo_gtrxl"].train(
                env=self.env,
                total_timesteps=100000,
                batch_size=config["AI"]["batch_size"],
                learning_rate=config["AI"]["learning_rate"],
                gradient_clip=config["AI"]["gradient_clip"]
            )
            
            # Entraînement du CNN-LSTM
            self.models["cnn_lstm"].train(
                X_train,
                y_train,
                epochs=config["AI"]["n_epochs"],
                batch_size=config["AI"]["batch_size"],
                validation_split=0.2
            )
            
            # Mise à jour du timestamp d'entraînement
            self.last_training_time = datetime.utcnow()
            
            # Sauvegarde des modèles
            self._save_models()
            
            logger.info(f"[{current_time}] ✅ Entraînement terminé avec succès")
            
        except Exception as e:
            logger.error(f"[{current_time}] ❌ Erreur lors de l'entraînement: {e}")
            raise

    def _prepare_training_data(self, historical_data, initial_analysis):
        """Prépare les données pour l'entraînement"""
        current_time = "2025-06-06 07:42:32"  # Mise à jour timestamp
        
        try:
            features = []
            labels = []
            
            # Pour chaque timeframe
            for timeframe in config["TRADING"]["timeframes"]:
                tf_data = historical_data[timeframe]
                tf_analysis = initial_analysis[timeframe]
                
                # Extraction des features
                technical_features = self._extract_technical_features(tf_data)
                market_features = self._extract_market_features(tf_data)
                indicator_features = self._extract_indicator_features(tf_analysis)
                
                # Combinaison des features
                combined_features = np.concatenate([
                    technical_features,
                    market_features,
                    indicator_features
                ], axis=1)
                
                features.append(combined_features)
                
                # Création des labels (returns futurs)
                future_returns = self._calculate_future_returns(tf_data)
                labels.append(future_returns)
            
            # Fusion des données de différents timeframes
            X = np.concatenate(features, axis=1)
            y = np.mean(labels, axis=0)
            
            return X, y
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur préparation données: {e}")
            raise

    def _extract_technical_features(self, data):
        """Extrait les features techniques des données"""
        current_time = "2025-06-06 07:42:32"  # Mise à jour timestamp
        
        try:
            features = []
            
            # Features de tendance
            if trend_data := self._calculate_trend_features(data):
                features.append(trend_data)
                
            # Features de momentum
            if momentum_data := self._calculate_momentum_features(data):
                features.append(momentum_data)
                
            # Features de volatilité
            if volatility_data := self._calculate_volatility_features(data):
                features.append(volatility_data)
                
            # Features de volume
            if volume_data := self._calculate_volume_features(data):
                features.append(volume_data)
                
            # Features d'orderflow
            if orderflow_data := self._calculate_orderflow_features(data):
                features.append(orderflow_data)
                
            return np.concatenate(features, axis=1)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur extraction features techniques: {e}")
            return np.array([])

    def _extract_market_features(self, data):
        """Extrait les features de marché"""
        current_time = "2025-06-06 07:42:32"  # Mise à jour timestamp
        
        try:
            features = []
            
            # Prix relatifs
            close = data['close'].values
            features.append(close[1:] / close[:-1] - 1)  # Returns
            
            # Volumes relatifs
            volume = data['volume'].values
            features.append(volume[1:] / volume[:-1] - 1)  # Volume change
            
            # Spread
            features.append((data['high'] - data['low']) / data['close'])
            
            # Gap analysis
            features.append(self._calculate_gap_features(data))
            
            # Liquidité
            features.append(self._calculate_liquidity_features(data))
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur extraction features marché: {e}")
            return np.array([])

    def _extract_indicator_features(self, analysis):
        """Extrait les features des indicateurs"""
        current_time = "2025-06-06 07:42:32"  # Mise à jour timestamp
        
        try:
            features = []
            
            # Features de tendance
            if "trend" in analysis:
                trend_strength = analysis["trend"].get("trend_strength", 0)
                features.append(trend_strength)
                
            # Features de volatilité
            if "volatility" in analysis:
                volatility = analysis["volatility"].get("current_volatility", 0)
                features.append(volatility)
                
            # Features de volume
            if "volume" in analysis:
                volume_profile = analysis["volume"].get("volume_profile", {})
                strength = float(volume_profile.get("strength", 0))
                features.append(strength)
                
            # Signal dominant
            if "dominant_signal" in analysis:
                signal_mapping = {
                    "Bullish": 1,
                    "Bearish": -1,
                    "Neutral": 0
                }
                signal = signal_mapping.get(analysis["dominant_signal"], 0)
                features.append(signal)
                
            return np.array(features)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur extraction features indicateurs: {e}")
            return np.array([])
    def _calculate_trend_features(self, data):
        """Calcule les features de tendance"""
        current_time = "2025-06-06 07:43:21"  # Mise à jour timestamp
        
        try:
            features = []
            
            # Supertrend
            if st_data := self._calculate_supertrend(data):
                features.append(st_data["value"])
                features.append(st_data["direction"])
                features.append(st_data["strength"])
            
            # Ichimoku
            if ichi_data := self._calculate_ichimoku(data):
                features.append(ichi_data["tenkan"] / data["close"])
                features.append(ichi_data["kijun"] / data["close"])
                features.append(ichi_data["senkou_a"] / data["close"])
                features.append(ichi_data["senkou_b"] / data["close"])
                features.append(ichi_data["cloud_strength"])
            
            # EMA Ribbon
            if ema_data := self._calculate_ema_ribbon(data):
                features.append(ema_data["trend"])
                features.append(ema_data["strength"])
                for ema in ema_data["emas"].values():
                    features.append(ema / data["close"])
            
            # Parabolic SAR
            if psar_data := self._calculate_psar(data):
                features.append(psar_data["value"] / data["close"])
                features.append(psar_data["trend"])
                features.append(psar_data["strength"])
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur calcul features tendance: {e}")
            return np.array([])

    def _calculate_momentum_features(self, data):
        """Calcule les features de momentum"""
        current_time = "2025-06-06 07:43:21"  # Mise à jour timestamp
        
        try:
            features = []
            
            # RSI
            if rsi_data := self._calculate_rsi(data):
                features.append(rsi_data["value"])
                features.append(float(rsi_data["overbought"]))
                features.append(float(rsi_data["oversold"]))
                features.append(rsi_data["divergence"])
            
            # Stochastic RSI
            if stoch_data := self._calculate_stoch_rsi(data):
                features.append(stoch_data["k_line"])
                features.append(stoch_data["d_line"])
                features.append(float(stoch_data["overbought"]))
                features.append(float(stoch_data["oversold"]))
                features.append(stoch_data["crossover"])
            
            # MACD
            if macd_data := self._calculate_macd(data):
                features.append(macd_data["macd"])
                features.append(macd_data["signal"])
                features.append(macd_data["histogram"])
                features.append(macd_data["crossover"])
                features.append(macd_data["strength"])
            
            # Awesome Oscillator
            if ao_data := self._calculate_ao(data):
                features.append(ao_data["value"])
                features.append(ao_data["momentum_shift"])
                features.append(ao_data["strength"])
                features.append(float(ao_data["zero_cross"]))
            
            # TSI
            if tsi_data := self._calculate_tsi(data):
                features.append(tsi_data["tsi"])
                features.append(tsi_data["signal"])
                features.append(tsi_data["histogram"])
                features.append(tsi_data["divergence"])
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur calcul features momentum: {e}")
            return np.array([])

    def _calculate_volatility_features(self, data):
        """Calcule les features de volatilité"""
        current_time = "2025-06-06 07:43:21"  # Mise à jour timestamp
        
        try:
            features = []
            
            # Bollinger Bands
            if bb_data := self._calculate_bbands(data):
                features.append((bb_data["upper"] - data["close"]) / data["close"])
                features.append((bb_data["middle"] - data["close"]) / data["close"])
                features.append((bb_data["lower"] - data["close"]) / data["close"])
                features.append(bb_data["bandwidth"])
                features.append(bb_data["percent_b"])
                features.append(float(bb_data["squeeze"]))
            
            # Keltner Channels
            if kc_data := self._calculate_keltner(data):
                features.append((kc_data["upper"] - data["close"]) / data["close"])
                features.append((kc_data["middle"] - data["close"]) / data["close"])
                features.append((kc_data["lower"] - data["close"]) / data["close"])
                features.append(kc_data["width"])
                features.append(kc_data["position"])
            
            # ATR
            if atr_data := self._calculate_atr(data):
                features.append(atr_data["value"])
                features.append(atr_data["normalized"])
                features.append(atr_data["trend"])
                features.append(atr_data["volatility_regime"])
            
            # VIX Fix
            if vix_data := self._calculate_vix_fix(data):
                features.append(vix_data["value"])
                features.append(vix_data["regime"])
                features.append(vix_data["trend"])
                features.append(vix_data["percentile"])
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur calcul features volatilité: {e}")
            return np.array([])

    def _calculate_gap_features(self, data):
        """Calcule les features de gaps"""
        current_time = "2025-06-06 07:43:21"  # Mise à jour timestamp
        
        try:
            features = []
            
            # Prix d'ouverture vs clôture précédente
            open_close_gap = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)
            features.append(open_close_gap)
            
            # Gap haussier/baissier
            features.append(np.where(open_close_gap > 0, 1, -1))
            
            # Force du gap
            features.append(abs(open_close_gap))
            
            # Gap comblé
            gap_filled = (data["low"] <= data["close"].shift(1)) & (data["high"] >= data["open"])
            features.append(gap_filled.astype(float))
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur calcul features gaps: {e}")
            return np.array([])
    def _calculate_liquidity_features(self, data):
        """Calcule les features de liquidité"""
        current_time = "2025-06-06 07:44:10"  # Mise à jour timestamp
        
        try:
            features = []
            
            # Analyse du carnet d'ordres
            if orderbook := self.buffer.get_orderbook(data.name):
                # Déséquilibre bid/ask
                bid_volume = sum(vol for _, vol in orderbook["bids"][:10])
                ask_volume = sum(vol for _, vol in orderbook["asks"][:10])
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                features.append(imbalance)
                
                # Profondeur de marché
                depth = (bid_volume + ask_volume) / data["volume"].mean()
                features.append(depth)
                
                # Spread relatif
                spread = (orderbook["asks"][0][0] - orderbook["bids"][0][0]) / orderbook["bids"][0][0]
                features.append(spread)
                
                # Clusters de liquidité
                clusters = self._detect_liquidity_clusters(orderbook)
                features.append(len(clusters["bid_clusters"]))
                features.append(len(clusters["ask_clusters"]))
                
                # Score de résistance à l'impact
                impact_resistance = self._calculate_impact_resistance(orderbook)
                features.append(impact_resistance)
            
            # Métriques historiques
            # Volume moyen sur 24h
            vol_24h = data["volume"].rolling(window=1440).mean()  # 1440 minutes = 24h
            features.append(data["volume"] / vol_24h)
            
            # Ratio de liquidité de Amihud
            daily_returns = data["close"].pct_change()
            amihud = abs(daily_returns) / (data["volume"] * data["close"])
            features.append(amihud)
            
            # Ratio de turnover
            turnover = data["volume"] * data["close"] / data["volume"].rolling(window=20).mean()
            features.append(turnover)
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur calcul features liquidité: {e}")
            return np.array([])

    def _detect_liquidity_clusters(self, orderbook):
        """Détecte les clusters de liquidité dans le carnet d'ordres"""
        current_time = "2025-06-06 07:44:10"  # Mise à jour timestamp
        
        try:
            bid_clusters = []
            ask_clusters = []
            
            # Paramètres de clustering
            min_volume = 1.0  # Volume minimum pour un cluster
            price_threshold = 0.001  # Distance maximale entre prix pour un même cluster
            
            # Détection des clusters côté bid
            current_cluster = {"start_price": None, "total_volume": 0}
            for price, volume in orderbook["bids"]:
                if volume >= min_volume:
                    if current_cluster["start_price"] is None:
                        current_cluster = {"start_price": price, "total_volume": volume}
                    elif abs(price - current_cluster["start_price"]) <= price_threshold:
                        current_cluster["total_volume"] += volume
                    else:
                        if current_cluster["total_volume"] >= min_volume:
                            bid_clusters.append(current_cluster)
                        current_cluster = {"start_price": price, "total_volume": volume}
            
            # Détection des clusters côté ask
            current_cluster = {"start_price": None, "total_volume": 0}
            for price, volume in orderbook["asks"]:
                if volume >= min_volume:
                    if current_cluster["start_price"] is None:
                        current_cluster = {"start_price": price, "total_volume": volume}
                    elif abs(price - current_cluster["start_price"]) <= price_threshold:
                        current_cluster["total_volume"] += volume
                    else:
                        if current_cluster["total_volume"] >= min_volume:
                            ask_clusters.append(current_cluster)
                        current_cluster = {"start_price": price, "total_volume": volume}
            
            return {
                "bid_clusters": bid_clusters,
                "ask_clusters": ask_clusters,
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur détection clusters: {e}")
            return {"bid_clusters": [], "ask_clusters": [], "timestamp": current_time}

    def _calculate_impact_resistance(self, orderbook, impact_size=1.0):
        """Calcule la résistance à l'impact de marché"""
        current_time = "2025-06-06 07:44:10"  # Mise à jour timestamp
        
        try:
            # Calcul de l'impact sur les bids
            cumulative_bid_volume = 0
            bid_impact = 0
            for price, volume in orderbook["bids"]:
                cumulative_bid_volume += volume
                if cumulative_bid_volume >= impact_size:
                    bid_impact = (orderbook["bids"][0][0] - price) / orderbook["bids"][0][0]
                    break
            
            # Calcul de l'impact sur les asks
            cumulative_ask_volume = 0
            ask_impact = 0
            for price, volume in orderbook["asks"]:
                cumulative_ask_volume += volume
                if cumulative_ask_volume >= impact_size:
                    ask_impact = (price - orderbook["asks"][0][0]) / orderbook["asks"][0][0]
                    break
            
            # Score de résistance
            resistance_score = 1 / (bid_impact + ask_impact) if (bid_impact + ask_impact) > 0 else float('inf')
            
            return resistance_score
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur calcul résistance impact: {e}")
            return 0.0

    def _calculate_future_returns(self, data, horizons=[1, 5, 10, 20]):
        """Calcule les returns futurs pour différents horizons"""
        current_time = "2025-06-06 07:44:10"  # Mise à jour timestamp
        
        try:
            returns = []
            
            for horizon in horizons:
                # Calcul du return futur
                future_return = data["close"].shift(-horizon) / data["close"] - 1
                returns.append(future_return)
                
                # Calcul de la volatilité future
                future_volatility = data["close"].rolling(window=horizon).std().shift(-horizon)
                returns.append(future_volatility)
                
                # Calcul du volume futur normalisé
                future_volume = (data["volume"].shift(-horizon) / data["volume"]).rolling(window=horizon).mean()
                returns.append(future_volume)
            
            return np.column_stack(returns)
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur calcul returns futurs: {e}")
            return np.array([])
    def _save_models(self):
        """Sauvegarde les modèles entraînés"""
        current_time = "2025-06-06 07:44:59"  # Mise à jour timestamp
        
        try:
            # Création du dossier de sauvegarde
            save_dir = os.path.join(current_dir, "models")
            os.makedirs(save_dir, exist_ok=True)
            
            # Sauvegarde du modèle hybride
            hybrid_path = os.path.join(save_dir, "hybrid_model.pt")
            torch.save(self.hybrid_model.state_dict(), hybrid_path)
            
            # Sauvegarde du PPO-GTrXL
            ppo_path = os.path.join(save_dir, "ppo_gtrxl.pt")
            torch.save(self.models["ppo_gtrxl"].state_dict(), ppo_path)
            
            # Sauvegarde du CNN-LSTM
            cnn_lstm_path = os.path.join(save_dir, "cnn_lstm.pt")
            torch.save(self.models["cnn_lstm"].state_dict(), cnn_lstm_path)
            
            # Sauvegarde des métadonnées
            metadata = {
                "timestamp": current_time,
                "user": self.current_user,
                "model_versions": {
                    "hybrid": self.hybrid_model.version,
                    "ppo_gtrxl": self.models["ppo_gtrxl"].version,
                    "cnn_lstm": self.models["cnn_lstm"].version
                },
                "training_metrics": self._get_training_metrics()
            }
            
            metadata_path = os.path.join(save_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"[{current_time}] ✅ Modèles sauvegardés avec succès")
            
        except Exception as e:
            logger.error(f"[{current_time}] ❌ Erreur sauvegarde modèles: {e}")
            raise

    def _get_training_metrics(self):
        """Récupère les métriques d'entraînement"""
        current_time = "2025-06-06 07:44:59"  # Mise à jour timestamp
        
        try:
            metrics = {
                "hybrid_model": {
                    "loss": self.hybrid_model.training_history["loss"],
                    "val_loss": self.hybrid_model.training_history["val_loss"],
                    "accuracy": self.hybrid_model.training_history["accuracy"]
                },
                "ppo_gtrxl": {
                    "policy_loss": self.models["ppo_gtrxl"].training_info["policy_loss"],
                    "value_loss": self.models["ppo_gtrxl"].training_info["value_loss"],
                    "entropy": self.models["ppo_gtrxl"].training_info["entropy"]
                },
                "cnn_lstm": {
                    "loss": self.models["cnn_lstm"].history["loss"],
                    "val_loss": self.models["cnn_lstm"].history["val_loss"],
                    "mae": self.models["cnn_lstm"].history["mae"]
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur récupération métriques: {e}")
            return {}

    async def _should_stop_trading(self):
        """Vérifie les conditions d'arrêt du trading"""
        current_time = "2025-06-06 07:44:59"  # Mise à jour timestamp
        
        try:
            # Vérification du circuit breaker
            if await self.circuit_breaker.should_stop_trading():
                logger.warning(f"[{current_time}] Circuit breaker activé")
                return True
            
            # Vérification du drawdown maximum
            current_drawdown = self.position_manager.calculate_drawdown()
            if current_drawdown > config["RISK"]["max_drawdown"]:
                logger.warning(f"[{current_time}] Drawdown maximum atteint: {current_drawdown:.2%}")
                return True
            
            # Vérification de la perte journalière
            daily_loss = self.position_manager.calculate_daily_loss()
            if daily_loss > config["RISK"]["daily_stop_loss"]:
                logger.warning(f"[{current_time}] Stop loss journalier atteint: {daily_loss:.2%}")
                return True
            
            # Vérification des conditions de marché
            market_conditions = await self._check_market_conditions()
            if not market_conditions["safe_to_trade"]:
                logger.warning(f"[{current_time}] Conditions de marché dangereuses: {market_conditions['reason']}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur vérification conditions d'arrêt: {e}")
            return True  # Par sécurité

    async def _check_market_conditions(self):
        """Vérifie les conditions de marché"""
        current_time = "2025-06-06 07:44:59"  # Mise à jour timestamp
        
        try:
            conditions = {
                "safe_to_trade": True,
                "reason": None
            }
            
            # Vérification de la volatilité
            volatility = self._analyze_volatility()
            if volatility["current"] > volatility["threshold"] * 2:
                conditions["safe_to_trade"] = False
                conditions["reason"] = "Volatilité excessive"
                return conditions
            
            # Vérification de la liquidité
            liquidity = await self._analyze_market_liquidity()
            if liquidity["status"] == "insufficient":
                conditions["safe_to_trade"] = False
                conditions["reason"] = "Liquidité insuffisante"
                return conditions
            
            # Vérification des news à haut risque
            if await self._check_high_risk_news():
                conditions["safe_to_trade"] = False
                conditions["reason"] = "News à haut risque"
                return conditions
            
            # Vérification des conditions techniques
            technical_check = self._check_technical_conditions()
            if not technical_check["safe"]:
                conditions["safe_to_trade"] = False
                conditions["reason"] = technical_check["reason"]
                return conditions
            
            return conditions
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur vérification conditions marché: {e}")
            return {"safe_to_trade": False, "reason": "Erreur système"}
    async def _analyze_market_liquidity(self):
        """Analyse détaillée de la liquidité du marché"""
        current_time = "2025-06-06 07:45:39"  # Mise à jour timestamp
        
        try:
            liquidity_status = {
                "status": "sufficient",
                "metrics": {},
                "timestamp": current_time
            }
            
            # Analyse du carnet d'ordres
            for pair in config["TRADING"]["pairs"]:
                orderbook = self.buffer.get_orderbook(pair)
                if orderbook:
                    # Profondeur de marché
                    depth = self._calculate_market_depth(orderbook)
                    
                    # Ratio bid/ask
                    bid_ask_ratio = self._calculate_bid_ask_ratio(orderbook)
                    
                    # Spread moyen
                    avg_spread = self._calculate_average_spread(orderbook)
                    
                    # Résistance à l'impact
                    impact_resistance = self._calculate_impact_resistance(orderbook)
                    
                    liquidity_status["metrics"][pair] = {
                        "depth": depth,
                        "bid_ask_ratio": bid_ask_ratio,
                        "avg_spread": avg_spread,
                        "impact_resistance": impact_resistance
                    }
                    
                    # Vérification des seuils
                    if (depth < 100000 or  # Exemple de seuil
                        abs(1 - bid_ask_ratio) > 0.2 or
                        avg_spread > 0.001 or
                        impact_resistance < 0.5):
                        liquidity_status["status"] = "insufficient"
                        
            return liquidity_status
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur analyse liquidité: {e}")
            return {"status": "insufficient", "metrics": {}, "timestamp": current_time}

    def _check_technical_conditions(self):
        """Vérifie les conditions techniques du marché"""
        current_time = "2025-06-06 07:45:39"  # Mise à jour timestamp
        
        try:
            conditions = {
                "safe": True,
                "reason": None,
                "details": {}
            }
            
            for pair in config["TRADING"]["pairs"]:
                pair_data = self.buffer.get_latest_ohlcv(pair)
                
                # Vérification des divergences
                divergences = self._check_divergences(pair_data)
                if divergences["critical"]:
                    conditions["safe"] = False
                    conditions["reason"] = f"Divergence critique sur {pair}"
                    conditions["details"][pair] = divergences
                    return conditions
                
                # Vérification des patterns critiques
                patterns = self._check_critical_patterns(pair_data)
                if patterns["detected"]:
                    conditions["safe"] = False
                    conditions["reason"] = f"Pattern critique sur {pair}: {patterns['pattern']}"
                    conditions["details"][pair] = patterns
                    return conditions
                
                # Vérification des niveaux clés
                levels = self._check_key_levels(pair_data)
                if levels["breach"]:
                    conditions["safe"] = False
                    conditions["reason"] = f"Rupture niveau clé sur {pair}"
                    conditions["details"][pair] = levels
                    return conditions
                
                conditions["details"][pair] = {
                    "divergences": divergences,
                    "patterns": patterns,
                    "levels": levels
                }
            
            return conditions
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur vérification technique: {e}")
            return {"safe": False, "reason": "Erreur système", "details": {}}

    def _check_divergences(self, data):
        """Détecte les divergences entre prix et indicateurs"""
        current_time = "2025-06-06 07:45:39"  # Mise à jour timestamp
        
        try:
            divergences = {
                "critical": False,
                "types": [],
                "timestamp": current_time
            }
            
            # RSI Divergence
            rsi = self._calculate_rsi(data)
            if rsi:
                price_peaks = self._find_peaks(data["close"])
                rsi_peaks = self._find_peaks(rsi["value"])
                
                if self._is_bearish_divergence(price_peaks, rsi_peaks):
                    divergences["critical"] = True
                    divergences["types"].append("RSI_BEARISH")
                
                if self._is_bullish_divergence(price_peaks, rsi_peaks):
                    divergences["types"].append("RSI_BULLISH")
            
            # MACD Divergence
            macd = self._calculate_macd(data)
            if macd:
                price_peaks = self._find_peaks(data["close"])
                macd_peaks = self._find_peaks(macd["histogram"])
                
                if self._is_bearish_divergence(price_peaks, macd_peaks):
                    divergences["critical"] = True
                    divergences["types"].append("MACD_BEARISH")
                
                if self._is_bullish_divergence(price_peaks, macd_peaks):
                    divergences["types"].append("MACD_BULLISH")
            
            return divergences
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur détection divergences: {e}")
            return {"critical": False, "types": [], "timestamp": current_time}

    def _check_critical_patterns(self, data):
        """Détecte les patterns techniques critiques"""
        current_time = "2025-06-06 07:45:39"  # Mise à jour timestamp
        
        try:
            patterns = {
                "detected": False,
                "pattern": None,
                "confidence": 0,
                "timestamp": current_time
            }
            
            # Head and Shoulders
            if self._detect_head_shoulders(data):
                patterns["detected"] = True
                patterns["pattern"] = "HEAD_AND_SHOULDERS"
                patterns["confidence"] = 0.85
                return patterns
            
            # Double Top/Bottom
            if self._detect_double_pattern(data):
                patterns["detected"] = True
                patterns["pattern"] = "DOUBLE_TOP" if data["close"].iloc[-1] < data["close"].mean() else "DOUBLE_BOTTOM"
                patterns["confidence"] = 0.80
                return patterns
            
            # Rising/Falling Wedge
            if self._detect_wedge(data):
                patterns["detected"] = True
                patterns["pattern"] = "RISING_WEDGE" if data["close"].iloc[-1] > data["close"].mean() else "FALLING_WEDGE"
                patterns["confidence"] = 0.75
                return patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"[{current_time}] Erreur détection patterns: {e}")
            return {"detected": False, "pattern": None, "confidence": 0, "timestamp": current_time}

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        st.set_page_config(layout="wide")
        st.title("Trading Bot Dashboard v4")
        
        import nest_asyncio
        nest_asyncio.apply()

        async def main():
            bot = TradingBotM4()
            with st.spinner("Exécution du bot en cours..."):
                await bot.run()

        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Bot arrêté manuellement")
        st.warning("Bot arrêté par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        st.error(f"Erreur critique: {str(e)}")