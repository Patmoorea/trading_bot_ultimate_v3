# 1. Imports système (DOIVENT ÊTRE EN PREMIER)
import os
import sys
import logging
import json
import plotly.graph_objects as go

# Ajout du chemin racine au PYTHONPATH
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Configuration environnement
os.environ['STREAMLIT_HIDE_PYTORCH_WARNING'] = '1'

# 3. Configuration asyncio et event loop
import asyncio
import nest_asyncio
from asyncio import AbstractEventLoop

def setup_event_loop() -> AbstractEventLoop:
    """Configure l'event loop pour Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply()
    return loop

# 4. Configuration Streamlit
import streamlit as st
st.set_page_config(
    page_title="Trading Bot Ultimate v4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 5. Imports standards
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from asyncio import TimeoutError
import telegram
from src.exchanges.binance_exchange import BinanceExchange
from src.portfolio.real_portfolio import RealPortfolio
import numpy as np
import ccxt
from dotenv import load_dotenv

# 6. Setup de l'event loop avant les imports PyTorch
setup_event_loop()

import ta

# 7. Imports ML/AI
import gymnasium as gym
from gymnasium import spaces
import torch
import pandas as pd

# 8. Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Imports des modules existants
from src.data.realtime.websocket.client import MultiStreamManager, StreamConfig
from src.core.buffer.circular_buffer import CircularBuffer
from src.indicators.advanced.multi_timeframe import MultiTimeframeAnalyzer, TimeframeConfig
from src.analysis.indicators.orderflow.orderflow_analysis import OrderFlowAnalysis, OrderFlowConfig
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

# Imports des indicateurs
from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators
from src.analysis.indicators.momentum.momentum import MomentumIndicators
from src.analysis.indicators.volatility.volatility import VolatilityIndicators
from src.analysis.indicators.volume.volume_analysis import VolumeAnalysis
from src.analysis.indicators.orderflow.orderflow_analysis import OrderFlowAnalysis, OrderFlowConfig
from src.analysis.indicators.trend.indicators import TrendIndicators

# Dans les imports, ajoutons les composants existants
from src.binance.binance_ws import AsyncClient, BinanceSocketManager
from src.connectors.binance import BinanceConnector
from src.exchanges.binance.binance_client import BinanceClient
from web_interface.app.services.news_analyzer import NewsAnalyzer

# Configuration
load_dotenv()
config = {
    "ARBITRAGE": {
        "exchanges": ["binance", "bitfinex", "kraken"],
        "min_profit": 0.001,
        "max_trade_size": 1000,
        "pairs": ["BTC/USDC", "ETH/USDC"],
        "timeout": 5,
        "volume_filter": 1000,
        "price_check": True,
        "max_slippage": 0.0005
    },
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
            return None

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
            return None

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
        self.news_analyzer = NewsAnalyzer()
    
        # Passage en mode réel
        self.trading_mode = "production"
        self.testnet = False

        # Activation des composants réels
        self.news_enabled = True
        self.arbitrage_enabled = True
        self.telegram_enabled = True

        # Configuration risque pour le réel
        self.max_drawdown = 0.05  # 5% max
        self.daily_stop_loss = 0.02  # 2% par jour
        self.max_position_size = 1000  # USDC

         # Récupération des variables d'environnement
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

        # Interface et monitoring
        self.dashboard = TradingDashboard()
        self.current_user = "Patmoorea"

        # Composants principaux
        self.arbitrage_engine = ArbitrageEngine(
            exchanges=config["ARBITRAGE"]["exchanges"],
            pairs=config["ARBITRAGE"]["pairs"],
            min_profit=config["ARBITRAGE"]["min_profit"],
            max_trade_size=config["ARBITRAGE"]["max_trade_size"],
            timeout=config["ARBITRAGE"]["timeout"],
            volume_filter=config["ARBITRAGE"]["volume_filter"],
            price_check=config["ARBITRAGE"]["price_check"],
            max_slippage=config["ARBITRAGE"]["max_slippage"]
        )
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

        self.binance_ws = AsyncClient.create(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET')
        )
        self.socket_manager = BinanceSocketManager(self.binance_ws)
        
        # Connecteur pour les orderbooks
        self.connector = BinanceConnector()
        
        # Client pour le spot trading
        self.spot_client = BinanceClient(
            api_key=os.getenv('BINANCE_API_KEY'),
            secret=os.getenv('BINANCE_API_SECRET')
        )
        
        # Exchange principal
        self.exchange = BinanceExchange(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=False
        )
      
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

    # Supprimez le dictionnaire self.indicators existant et remplacez-le par :
    def add_indicators(self, df):
        """Ajoute tous les indicateurs (130+) au DataFrame"""
        try:
            # Ajout de tous les indicateurs techniques
            df_with_indicators = ta.add_all_ta_features(
                df,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=True
            )
        
            # Organisez les indicateurs par catégories
            indicators = {
                'trend': {
                    'sma_fast': df_with_indicators['trend_sma_fast'],
                    'sma_slow': df_with_indicators['trend_sma_slow'],
                    'ema_fast': df_with_indicators['trend_ema_fast'],
                    'ema_slow': df_with_indicators['trend_ema_slow'],
                    'adx': df_with_indicators['trend_adx'],
                    'adx_pos': df_with_indicators['trend_adx_pos'],
                    'adx_neg': df_with_indicators['trend_adx_neg'],
                    'vortex_ind_pos': df_with_indicators['trend_vortex_ind_pos'],
                    'vortex_ind_neg': df_with_indicators['trend_vortex_ind_neg'],
                    'vortex_ind_diff': df_with_indicators['trend_vortex_ind_diff'],
                    'trix': df_with_indicators['trend_trix'],
                    'mass_index': df_with_indicators['trend_mass_index'],
                    'cci': df_with_indicators['trend_cci'],
                    'dpo': df_with_indicators['trend_dpo'],
                    'kst': df_with_indicators['trend_kst'],
                    'kst_sig': df_with_indicators['trend_kst_sig'],
                    'kst_diff': df_with_indicators['trend_kst_diff'],
                    'ichimoku_a': df_with_indicators['trend_ichimoku_a'],
                    'ichimoku_b': df_with_indicators['trend_ichimoku_b'],
                    'visual_ichimoku_a': df_with_indicators['trend_visual_ichimoku_a'],
                    'visual_ichimoku_b': df_with_indicators['trend_visual_ichimoku_b'],
                    'aroon_up': df_with_indicators['trend_aroon_up'],
                    'aroon_down': df_with_indicators['trend_aroon_down'],
                    'aroon_ind': df_with_indicators['trend_aroon_ind']
                },
                'momentum': {
                    'rsi': df_with_indicators['momentum_rsi'],
                    'stoch': df_with_indicators['momentum_stoch'],
                    'stoch_signal': df_with_indicators['momentum_stoch_signal'],
                    'tsi': df_with_indicators['momentum_tsi'],
                    'uo': df_with_indicators['momentum_uo'],
                    'stoch_rsi': df_with_indicators['momentum_stoch_rsi'],
                    'stoch_rsi_k': df_with_indicators['momentum_stoch_rsi_k'],
                    'stoch_rsi_d': df_with_indicators['momentum_stoch_rsi_d'],
                    'williams_r': df_with_indicators['momentum_wr'],
                    'ao': df_with_indicators['momentum_ao']
                },
                'volatility': {
                    'bbm': df_with_indicators['volatility_bbm'],
                    'bbh': df_with_indicators['volatility_bbh'],
                    'bbl': df_with_indicators['volatility_bbl'],
                    'bbw': df_with_indicators['volatility_bbw'],
                    'bbp': df_with_indicators['volatility_bbp'],
                    'kcc': df_with_indicators['volatility_kcc'],
                    'kch': df_with_indicators['volatility_kch'],
                    'kcl': df_with_indicators['volatility_kcl'],
                    'kcw': df_with_indicators['volatility_kcw'],
                    'kcp': df_with_indicators['volatility_kcp'],
                    'atr': df_with_indicators['volatility_atr'],
                    'ui': df_with_indicators['volatility_ui']
                },
                'volume': {
                    'mfi': df_with_indicators['volume_mfi'],
                    'adi': df_with_indicators['volume_adi'],
                    'obv': df_with_indicators['volume_obv'],
                    'cmf': df_with_indicators['volume_cmf'],
                    'fi': df_with_indicators['volume_fi'],
                    'em': df_with_indicators['volume_em'],
                    'sma_em': df_with_indicators['volume_sma_em'],
                    'vpt': df_with_indicators['volume_vpt'],
                    'nvi': df_with_indicators['volume_nvi'],
                    'vwap': df_with_indicators['volume_vwap']
                },
                'others': {
                    'dr': df_with_indicators['others_dr'],
                    'dlr': df_with_indicators['others_dlr'],
                    'cr': df_with_indicators['others_cr']
                }
            }
        
            logger.info(f"✅ Indicateurs calculés avec succès pour {len(indicators)} catégories")
            return indicators
        
        except Exception as e:
            logger.error(f"❌ Erreur calcul indicateurs: {e}")
            return None
    
    async def setup_streams(self):
        """Configure les streams de données en temps réel"""
        try:
            streams = []
            
            # Stream de trades pour chaque paire
            for pair in config["TRADING"]["pairs"]:
                symbol = pair.replace('/', '').lower()
                trade_socket = self.socket_manager.trade_socket(symbol)
                streams.append(trade_socket)
                
                # Stream d'orderbook
                depth_socket = self.socket_manager.depth_socket(symbol)
                streams.append(depth_socket)
                
                # Stream de klines (bougies)
                kline_socket = self.socket_manager.kline_socket(symbol, '1m')
                streams.append(kline_socket)
                
            # Démarrage des streams
            for stream in streams:
                asyncio.create_task(self._handle_stream(stream))
                
            logger.info("✅ Streams configurés avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur configuration streams: {e}")
            return None
            raise

    async def _handle_stream(self, stream):
        """Gère un stream de données"""
        try:
            async with stream as tscm:
                while True:
                    msg = await tscm.recv()
                    await self._process_stream_message(msg)
        except Exception as e:
            logger.error(f"Erreur stream: {e}")
            return None

    async def _process_stream_message(self, msg):
        """Traite les messages des streams"""
        try:
            if not msg:
                logger.warning("Message vide reçu")
                return
            
            if msg.get('e') == 'trade':
                await self._handle_trade(msg)
            elif msg.get('e') == 'depthUpdate':
                await self._handle_orderbook(msg)
            elif msg.get('e') == 'kline':
                await self._handle_kline(msg)
                
        except Exception as e:
            logger.error(f"Erreur traitement message: {e}")
            return None
         
    async def _handle_trade(self, msg):
        """Traite un trade"""
        try:
            trade_data = {
                'symbol': msg['s'],
                'price': float(msg['p']),
                'quantity': float(msg['q']),
                'time': msg['T'],
                'buyer': msg['b'],
                'seller': msg['a']
            }
            
            # Mise à jour du buffer
            self.buffer.update_trades(trade_data)
            
            # Analyse du volume
            self.volume_analysis.update(trade_data)
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Erreur traitement trade: {e}")
            return None

    async def _handle_orderbook(self, msg):
        """Traite une mise à jour d'orderbook"""
        try:
            orderbook_data = {
                'symbol': msg['s'],
                'bids': [[float(p), float(q)] for p, q in msg['b']],
                'asks': [[float(p), float(q)] for p, q in msg['a']],
                'time': msg['T']
            }
            
            # Mise à jour du buffer
            self.buffer.update_orderbook(orderbook_data)
            
            # Analyse de la liquidité
            await self._analyze_market_liquidity()
            
            return orderbook_data
            
        except Exception as e:
            logger.error(f"Erreur traitement orderbook: {e}")
            return None
        
        def decision_model(self, features, timestamp=None):
            try:
                policy = self.models["ppo_gtrxl"].get_policy(features)
                value = self.models["ppo_gtrxl"].get_value(features)
                return policy, value
            except Exception as e:
                logger.error(f"[{timestamp}] Erreur decision_model: {e}")
                return None, None
        
    async def _handle_kline(self, msg):
        """Traite une bougie"""
        try:
            kline = msg['k']
            kline_data = {
                'symbol': msg['s'],
                'interval': kline['i'],
                'time': kline['t'],
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'closed': kline['x']
            }
            
            # Mise à jour du buffer
            self.buffer.update_klines(kline_data)
            
            # Analyse technique si la bougie est fermée
            if kline_data['closed']:
                await self.analyze_signals(
                    market_data=self.buffer.get_latest_ohlcv(kline_data['symbol']),
                    indicators=self.advanced_indicators.analyze_timeframe(kline_data)
                )
            return kline_data

        except Exception as e:
            logger.error(f"Erreur traitement kline: {e}")
            return None
        
    def _add_risk_management(self, decision, timestamp=None):
        try:
            # Calcul du stop loss
            stop_loss = self._calculate_stop_loss(decision)
        
            # Calcul du take profit
            take_profit = self._calculate_take_profit(decision)
        
            # Ajout trailing stop
            trailing_stop = {
                "activation_price": stop_loss * 1.02,
                "callback_rate": 0.01
            }
        
            decision.update({
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop
            })
        
            return decision
        
        except Exception as e:
            logger.error(f"[{timestamp}] Erreur risk management: {e}")
            return decision

    async def get_latest_data(self):
        """Récupère les dernières données de marché"""
        try:
            data = {}
            for pair in config["TRADING"]["pairs"]:
                data[pair] = {}
                try:
                    # WebSocket prices
                    prices = await self.binance_ws.get_price(pair)
                    data[pair]['price'] = prices

                    # OrderBook
                    orderbook = await self.connector.get_order_book(pair)
                    data[pair]['orderbook'] = {
                        'bids': orderbook[0],  # Best bid
                        'asks': orderbook[1]   # Best ask
                    }

                    # Account data
                    account = await self.exchange.get_balance()
                    data[pair]['account'] = account
                except Exception as inner_e:
                    logger.error(f"Erreur pour {pair}: {inner_e}")
                    continue

            # Store in buffer
            if data:
                self.buffer.update_data(data)
                return data
            return None

        except Exception as e:
            logger.error(f"Erreur get_latest_data: {e}")
            return None

    async def study_market(self, period="7d"):
        """Analyse initiale du marché"""
        logger.info("🔊 Étude du marché en cours...")

        try:
            # Récupération des données historiques
            historical_data = await self.exchange.get_historical_data(
                config["TRADING"]["pairs"],
                config["TRADING"]["timeframes"],
                period
            )

            if not historical_data:
                raise ValueError("Données historiques non disponibles")

            # Analyse des indicateurs par timeframe
            indicators_analysis = {}
            for timeframe in config["TRADING"]["timeframes"]:
                try:
                    tf_data = historical_data[timeframe]
                    result = self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
                    indicators_analysis[timeframe] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Neutre"
                    } if result is None else result
                except Exception as tf_error:
                    logger.error(f"Erreur analyse timeframe {timeframe}: {tf_error}")
                    indicators_analysis[timeframe] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Erreur"
                    }

            # Détection du régime de marché
            regime = self.regime_detector.predict(indicators_analysis)
            logger.info(f"🔈 Régime de marché détecté: {regime}")

            # Génération et envoi du rapport
            try:
                analysis_report = self._generate_analysis_report(
                    indicators_analysis,
                    regime,
                )
                await self.telegram.send_message(analysis_report)
            except Exception as report_error:
                logger.error(f"Erreur génération rapport: {report_error}")

            # Mise à jour du dashboard
            try:
                self.dashboard.update_market_analysis(
                    historical_data=historical_data,
                    indicators=indicators_analysis,
                    regime=regime,
                )
            except Exception as dash_error:
                logger.error(f"Erreur mise à jour dashboard: {dash_error}")

            return regime, historical_data, indicators_analysis

        except Exception as e:
            logger.error(f"Erreur study_market: {e}")
            raise

    async def analyze_signals(self, market_data):
        """Analyse des signaux de trading basée sur tous les indicateurs"""
        try:
            # Obtention des indicateurs
            indicators = self.add_indicators(market_data)
            if not indicators:
                return None
            
            # Analyse des tendances
            trend_analysis = {
                'primary_trend': 'bullish' if indicators['trend']['ema_fast'] > indicators['trend']['sma_slow'] else 'bearish',
                'trend_strength': indicators['trend']['adx'],
                'trend_direction': 1 if indicators['trend']['vortex_ind_diff'] > 0 else -1,
                'ichimoku_signal': 'buy' if indicators['trend']['ichimoku_a'] > indicators['trend']['ichimoku_b'] else 'sell'
            }
        
            # Analyse du momentum
            momentum_analysis = {
                'rsi_signal': 'oversold' if indicators['momentum']['rsi'] < 30 else 'overbought' if indicators['momentum']['rsi'] > 70 else 'neutral',
                'stoch_signal': 'buy' if indicators['momentum']['stoch_rsi_k'] > indicators['momentum']['stoch_rsi_d'] else 'sell',
                'ultimate_signal': 'buy' if indicators['momentum']['uo'] > 70 else 'sell' if indicators['momentum']['uo'] < 30 else 'neutral'
            }
        
            # Analyse de la volatilité
            volatility_analysis = {
                'bb_signal': 'oversold' if market_data['close'].iloc[-1] < indicators['volatility']['bbl'].iloc[-1] else 'overbought' if market_data['close'].iloc[-1] > indicators['volatility']['bbh'].iloc[-1] else 'neutral',
                'kc_signal': 'breakout' if market_data['close'].iloc[-1] > indicators['volatility']['kch'].iloc[-1] else 'breakdown' if market_data['close'].iloc[-1] < indicators['volatility']['kcl'].iloc[-1] else 'range',
                'atr_volatility': indicators['volatility']['atr'].iloc[-1]
            }
        
            # Analyse du volume
            volume_analysis = {
                'mfi_signal': 'buy' if indicators['volume']['mfi'].iloc[-1] < 20 else 'sell' if indicators['volume']['mfi'].iloc[-1] > 80 else 'neutral',
                'cmf_trend': 'positive' if indicators['volume']['cmf'].iloc[-1] > 0 else 'negative',
                'obv_trend': 'up' if indicators['volume']['obv'].diff().iloc[-1] > 0 else 'down'
            }
        
            # Décision finale
            signal = {
                'timestamp': pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'volatility': volatility_analysis,
                'volume': volume_analysis,
                'recommendation': self._generate_recommendation(trend_analysis, momentum_analysis, volatility_analysis, volume_analysis)
            }
        
            logger.info(f"✅ Analyse des signaux complétée: {signal['recommendation']}")
            return signal
        
        except Exception as e:
            logger.error(f"❌ Erreur analyse signaux: {e}")
            return None

    def _generate_recommendation(self, trend, momentum, volatility, volume):
        """Génère une recommandation basée sur l'analyse des indicateurs"""
        try:
            # Système de points pour la décision
            points = 0
        
            # Points basés sur la tendance
            if trend['primary_trend'] == 'bullish': points += 2
            if trend['trend_strength'] > 25: points += 1
            if trend['trend_direction'] == 1: points += 1
        
            # Points basés sur le momentum
            if momentum['rsi_signal'] == 'oversold': points += 2
            if momentum['stoch_signal'] == 'buy': points += 1
            if momentum['ultimate_signal'] == 'buy': points += 1
        
            # Points basés sur la volatilité
            if volatility['bb_signal'] == 'oversold': points += 1
            if volatility['kc_signal'] == 'breakout': points += 1
        
            # Points basés sur le volume
            if volume['mfi_signal'] == 'buy': points += 1
            if volume['cmf_trend'] == 'positive': points += 1
            if volume['obv_trend'] == 'up': points += 1
        
            # Génération de la recommandation
            if points >= 8:
                return {'action': 'strong_buy', 'confidence': points/12}
            elif points >= 6:
                return {'action': 'buy', 'confidence': points/12}
            elif points <= 2:
                return {'action': 'strong_sell', 'confidence': 1 - points/12}
            elif points <= 4:
                return {'action': 'sell', 'confidence': 1 - points/12}
            else:
                return {'action': 'neutral', 'confidence': 0.5}
            
        except Exception as e:
            logger.error(f"❌ Erreur génération recommandation: {e}")
            return {'action': 'error', 'confidence': 0}

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
            logger.error(f"Erreur: {e}")
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
        # Vérification du circuit breaker
        if await self.circuit_breaker.should_stop_trading():
            await self.telegram.send_message(
                "⚠️ Trading suspendu: Circuit breaker activé\n"
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
                logger.error(f"Erreur: {e}")
                await self.telegram.send_message(
                    f"⚠️ Erreur d'exécution: {str(e)}\n"
                    f"Trader: {self.current_user}"
                )

    def _validate_trade(self, decision, position_size):
        """Validation finale avant l'exécution du trade"""
        try:
            # Vérification de la taille minimale
            if position_size < 0.001:  # Exemple de taille minimale
                return False

            # Vérification du spread
            if self._check_spread_too_high(decision["symbol"]):
                return False

            # Vérification de la liquidité
            if not self._check_sufficient_liquidity(decision["symbol"], position_size):
                return False

            # Vérification des news à haut risque
            if self._check_high_risk_news():
                return False

            # Vérification des limites de position
            if not self.position_manager.check_position_limits(position_size):
                return False

            # Vérification du timing d'entrée
            if not self._check_entry_timing(decision):
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur: {e}")
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
            logger.error(f"Erreur: {e}")
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
            logger.error(f"Erreur: {e}")
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
            logger.error(f"Erreur: {e}")
            return False

    def _analyze_momentum_signals(self):
        """Analyse des signaux de momentum"""

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
            }

        except Exception as e:
            logger.error(f"Erreur: {e}")

    def _analyze_volatility(self):
        """Analyse de la volatilité actuelle"""

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
                "indicators": {
                    "bbands": bbands,
                    "atr": atr
                }
            }

        except Exception as e:
            logger.error(f"Erreur: {e}")

class TradingBotM4:
    def _analyze_volume_profile(self):
        """Analyse du profil de volume"""
        try:
            vp = self._calculate_vp(self.buffer.get_latest())

            if not vp:
                return {
                    "supports_entry": False,
                    "poc": None,
                    "volume_trend": "insufficient_data"
                }

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
            }

        except Exception as e:
            logger.error(f"Erreur: {e}")
            return None
            
    async def run(self):
        """Méthode principale d'exécution"""
        try:
            logger.info(f"""
╔═════════════════════════════════════════════════════════════╗
║                Trading Bot Ultimate v4 Started               ║
╠═════════════════════════════════════════════════════════════╣
║ User: {self.current_user}                                   ║
║ Mode: {'REAL' if not self.testnet else 'TEST'}             ║
║ Status: INITIALIZING                                        ║
╚═════════════════════════════════════════════════════════════╝
""")
            # Initialisation
            await self.initialize()
            
            # Étude initiale du marché
            regime, historical_data, analysis = await self.study_market("7d")
            
            # Boucle principale
            while True:
                try:
                    # Récupération des données
                    latest_data = await self.get_latest_data()
                    if latest_data is None:
                        continue
                        
                    # Analyse des signaux
                    decision = await self.analyze_signals(
                        latest_data,
                        await self.calculate_indicators(latest_data)
                    )
                    
                    if decision and decision.get('should_trade', False):
                        # Exécution du trade
                        await self.execute_trades(decision)
                        
                    # Mise à jour du dashboard
                    await self.update_real_dashboard()
                    
                    # Délai avant prochaine itération
                    await asyncio.sleep(1)
                    
                except Exception as loop_error:
                    logger.error(f"Erreur dans la boucle: {loop_error}")
                    await asyncio.sleep(5)
                    continue
                    
        except KeyboardInterrupt:
            logger.info("❌ Arrêt manuel demandé")
            await self.shutdown()
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
            if hasattr(self, 'telegram'):
                await self.telegram.send_message(
                    chat_id=self.chat_id,
                    text=f"🚨 Erreur critique - Bot arrêté: {str(e)}"
                )
            raise
        finally:
            await self.shutdown()
            
    async def process_market_data(self):
        """Traite les données de marché en temps réel"""
        try:
            latest_data = await self.get_latest_data()
            if not latest_data:
                logger.warning("Pas de données disponibles")
                return None, None

            # Analyse des indicateurs
            indicators = {}
            for timeframe in config["TRADING"]["timeframes"]:
                if timeframe_data := latest_data.get(timeframe):
                    indicators[timeframe] = self.advanced_indicators.analyze_timeframe(
                        timeframe_data,
                        timeframe
                    )

            return latest_data, indicators

        except Exception as e:
            logger.error(f"Erreur process_market_data: {e}")
            return None, None

    def _should_train(self, historical_data):
        """Détermine si les modèles doivent être réentraînés"""
        try:
            # Vérification de la taille minimale des données
            if len(historical_data.get('1h', [])) < config["AI"]["min_training_size"]:
                return False

            # Vérification de la dernière session d'entraînement
                return True

            return time_since_training.days >= 1  # Réentraînement quotidien

        except Exception as e:
            logger.error(f"Erreur: {e}")
            return False

    async def _train_models(self, historical_data, initial_analysis):
        """Entraîne ou met à jour les modèles"""

        try:

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

            # Sauvegarde des modèles
            self._save_models()


        except Exception as e:
            logger.error(f"Erreur: {e}")
            raise

    def _prepare_training_data(self, historical_data, initial_analysis):
        """Prépare les données pour l'entraînement"""

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
            logger.error(f"Erreur: {e}")
            raise

    def _extract_technical_features(self, data):
        """Extrait les features techniques des données"""

        try:
            features = []

            # Features de tendance
            trend_data = self._calculate_trend_features(data)
            if trend_data:
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
            logger.error(f"Erreur: {e}")
            return np.array([])

    def _extract_market_features(self, data):
        """Extrait les features de marché"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])

    def _extract_indicator_features(self, analysis):
        """Extrait les features des indicateurs"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])
    
    def _calculate_trend_features(self, data):
        """Calcule les features de tendance"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])

    def _calculate_momentum_features(self, data):
        """Calcule les features de momentum"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])

    def _calculate_volatility_features(self, data):
        """Calcule les features de volatilité"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])

    def _calculate_gap_features(self, data):
        """Calcule les features de gaps"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _calculate_liquidity_features(self, data):
        """Calcule les features de liquidité"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])

    def _detect_liquidity_clusters(self, orderbook):
        """Détecte les clusters de liquidité dans le carnet d'ordres"""

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
            }

        except Exception as e:
            logger.error(f"Erreur: {e}")

    def _calculate_impact_resistance(self, orderbook, impact_size=1.0):
        """Calcule la résistance à l'impact de marché"""

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
            logger.error(f"Erreur: {e}")
            return

    def _calculate_future_returns(self, data, horizons=[1, 5, 10, 20]):
        """Calcule les returns futurs pour différents horizons"""

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
            logger.error(f"Erreur: {e}")
            return np.array([])
    def _save_models(self):
        """Sauvegarde les modèles entraînés"""

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


        except Exception as e:
            logger.error(f"Erreur: {e}")
            raise

    def _get_training_metrics(self):
        """Récupère les métriques d'entraînement"""

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
            logger.error(f"Erreur: {e}")
            return {}

    async def _should_stop_trading(self):
        """Vérifie les conditions d'arrêt du trading"""

        try:
            # Vérification du circuit breaker
            if await self.circuit_breaker.should_stop_trading():
                return True

            # Vérification du drawdown maximum
            current_drawdown = self.position_manager.calculate_drawdown()
            if current_drawdown > config["RISK"]["max_drawdown"]:
                return True

            # Vérification de la perte journalière
            daily_loss = self.position_manager.calculate_daily_loss()
            if daily_loss > config["RISK"]["daily_stop_loss"]:
                return True

            # Vérification des conditions de marché
            market_conditions = await self._check_market_conditions()
            if not market_conditions["safe_to_trade"]:
                return True

            return False

        except Exception as e:
            logger.error(f"Erreur: {e}")
            return True  # Par sécurité

    async def _check_market_conditions(self):
        """Vérifie les conditions de marché"""

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
            logger.error(f"Erreur: {e}")
            return {"safe_to_trade": False, "reason": "Erreur système"}
    async def _analyze_market_liquidity(self):
        """Analyse détaillée de la liquidité du marché"""
        try:
            liquidity_status = {
                "status": "sufficient",
                "metrics": {},
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
            logger.error(f"Erreur analyse liquidité: {e}")
            return {"status": "insufficient", "metrics": {}}

    def _check_technical_conditions(self):
        """Vérifie les conditions techniques du marché"""

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
            logger.error(f"Erreur: {e}")
            return {"safe": False, "reason": "Erreur système", "details": {}}

    def _check_divergences(self, data):
        """Détecte les divergences entre prix et indicateurs"""

        try:
            divergences = {
                "critical": False,
                "types": [],
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
            logger.error(f"Erreur: {e}")

    def _check_critical_patterns(self, data):
        """Détecte les patterns techniques critiques"""

        try:
            patterns = {
                "detected": False,
                "pattern": None,
                "confidence": 0,
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
            logger.error(f"Erreur: {e}")

async def run_trading_bot():
    """Point d'entrée synchrone pour le bot de trading"""
    try:
        # Interface Streamlit
        st.title("Trading Bot Ultimate v4 🤖")

        # Informations de session
        st.sidebar.info("""
        **Session Info**
        """)
        
        # Initialisation des valeurs par défaut
        portfolio_value = 0.0
        pnl = 0.0
        
        # Configuration trading
        with st.sidebar:
            st.header("Trading Configuration")
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Medium"
            )
            pairs = st.multiselect(
                "Trading Pairs",
                options=config["TRADING"]["pairs"],
                default=config["TRADING"]["pairs"]
            )

        # Stats en temps réel
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", f"{portfolio_value:.2f} USDC", f"{pnl:+.2f} USDC")
        with col2:
            st.metric("Active Positions", "2", "Open")
        with col3:
            st.metric("24h P&L", "+123 USDC", "+1.23%")

        # Bouton de démarrage
        if st.button("Start Trading Bot", type="primary"):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                with st.spinner("Initialisation du bot de trading..."):
                    bot = TradingBotM4()
                    loop.run_until_complete(bot.run())
            except Exception as e:
                logger.error(f"Erreur du bot: {e}")
                st.error(f"Bot error: {str(e)}")
                logging.error("Bot error", exc_info=True)
            finally:
                loop.close()

    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        st.error(f"Critical error: {str(e)}")
        logging.error("Fatal error", exc_info=True)

# Ajout des méthodes de trading réel à la classe TradingBotM4
async def setup_real_exchange(self):
    """Configuration sécurisée de l'exchange"""
    if not hasattr(self, 'exchange') or self.exchange is None:
        try:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_API_SECRET'),
                'enableRateLimit': True
            })
            await self.exchange.load_markets()
            logger.info("Exchange configuré avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur configuration exchange: {e}")
            return False

async def setup_real_telegram(self):
    """Configuration sécurisée de Telegram"""
    if not hasattr(self, 'telegram') or self.telegram is None:
        try:
            self.telegram = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
            self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
            await self.telegram.send_message(
                chat_id=self.chat_id,
            )
            return True
        except Exception as e:
            logger.error(f"Erreur configuration Telegram: {e}")
            return None
            return False

async def get_real_portfolio(self):
    """Récupération sécurisée du portfolio"""
    try:
        balance = await self.exchange.fetch_balance()
        positions = await self.exchange.fetch_positions()
        
        portfolio = {
            'total_value': float(balance['total'].get('USDC', 0)),
            'free': float(balance['free'].get('USDC', 0)),
            'used': float(balance['used'].get('USDC', 0)),
            'positions': [
                {
                    'symbol': pos['symbol'],
                    'size': pos['contracts'],
                    'value': pos['notional'],
                    'pnl': pos['unrealizedPnl']
                }
                for pos in positions if pos['contracts'] > 0
            ]
        }

        await self.telegram.send_message(
            chat_id=self.chat_id,
            text=f"""💰 Portfolio Update:
Total: {portfolio['total_value']:.2f} USDC
Positions: {len(portfolio['positions'])}
PnL: {sum(p['pnl'] for p in portfolio['positions']):.2f} USDC"""
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Erreur portfolio: {e}")
        return None

async def execute_real_trade(self, signal):
    """Exécution sécurisée des trades"""
    try:
        # Vérification du solde
        balance = await self.get_real_portfolio()
        if not balance or balance['free'] < signal['amount'] * signal['price']:
            logger.warning("Solde insuffisant pour le trade")
            return None
            
        # Calcul stop loss et take profit
        stop_loss = signal['price'] * (1 - signal['risk_ratio'])
        take_profit = signal['price'] * (1 + signal['risk_ratio'] * 2)
        
        # Placement de l'ordre
        order = await self.exchange.create_order(
            symbol=signal['symbol'],
            type='limit',
            side=signal['side'],
            amount=signal['amount'],
            price=signal['price'],
            params={
                'stopLoss': {
                    'type': 'trailing',
                    'stopPrice': stop_loss,
                    'callbackRate': 1.0
                },
                'takeProfit': {
                    'price': take_profit
                }
            }
        )
        
        # Notification
        await self.telegram.send_message(
            chat_id=self.chat_id,
            text=f"""🔵 Nouvel ordre:
Symbol: {order['symbol']}
Type: {order['type']}
Side: {order['side']}
Amount: {order['amount']}
Prix: {order['price']}
Stop Loss: {stop_loss}
Take Profit: {take_profit}"""
        )
        
        return order
    
    except Exception as e:  # Correction de l'indentation ici
        logger.error(f"Erreur trade: {e}")
        return None
        
# Extension sécurisée de la méthode run() existante
async def run_real_trading(self):
    """Boucle de trading réel sécurisée"""
    try:
        # Initialisation des connexions réelles
        if not await self.setup_real_exchange():
            raise Exception("Échec configuration exchange")
            
        if not await self.setup_real_telegram():
            raise Exception("Échec configuration Telegram")
            
        # Démarrage du bot
        logger.info(f"""
╔═════════════════════════════════════════════════════════════╗
║                Trading Bot Ultimate v4 - REAL               ║
╠═════════════════════════════════════════════════════════════╣
║ User: {self.current_user}                                  ║
║ Mode: REAL TRADING                                         ║
║ Status: RUNNING                                            ║
╚═════════════════════════════════════════════════════════════╝
        """)
        
        # Premier check du portfolio
        initial_portfolio = await self.get_real_portfolio()
        if not initial_portfolio:
            raise Exception("Impossible de récupérer le portfolio")
            
        # Boucle principale
        while True:
            try:
                # Analyse et décision
                decision = await self.analyze_signals(
                    await self.get_latest_data(),
                    await self.calculate_indicators()
                )
                
                if decision and decision.get('should_trade', False):
                    # Exécution réelle
                    trade_result = await self.execute_real_trade(decision)
                    if trade_result:
                        logger.info(f"Trade exécuté: {trade_result['id']}")
                        
                # Mise à jour portfolio
                await self.get_real_portfolio()
                
                # Délai avant prochaine itération
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle: {e}")
                await asyncio.sleep(5)
                continue
                
    except Exception as e:
        logger.error(f"Erreur: {e}")
        if hasattr(self, 'telegram'):
            await self.telegram.send_message(
                chat_id=self.chat_id,
                text=f"🚨 Erreur critique - Bot arrêté: {str(e)}"
            )
        raise




# Modification de la fonction update_dashboard pour utiliser les vraies données
async def update_real_dashboard(self):
    """Met à jour le dashboard avec les données réelles"""
    try:
        portfolio = RealPortfolio()
        if await portfolio.update(self.exchange):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Portfolio Value",
                    f"{portfolio.portfolio_value:.2f} USDC",
                    f"{portfolio.daily_pnl:+.2f} USDC"
                )
            with col2:
                st.metric(
                    "Active Positions",
                    str(portfolio.positions_count)
                )
            with col3:
                pnl_percent = (portfolio.daily_pnl / portfolio.portfolio_value * 100) if portfolio.portfolio_value > 0 else 0
                st.metric(
                    "24h P&L",
                    f"{portfolio.daily_pnl:+.2f} USDC",
                    f"{pnl_percent:+.2f}%"
                )
    except Exception as e:
        logger.error(f"Erreur mise à jour dashboard: {e}")
        st.error(f"Erreur mise à jour métriques: {str(e)}")

    def _get_portfolio_value(self):
        """Récupère la valeur actuelle du portfolio"""
        try:
            if hasattr(self, 'position_manager') and hasattr(self.position_manager, 'positions'):
                return sum(self.position_manager.positions.values())
            return 0.0
        except Exception as e:
            logger.error(f"Erreur calcul portfolio: {e}")
            return None

    def _calculate_total_pnl(self):
        """Calcule le PnL total"""
        try:
            if hasattr(self, 'position_history'):
                return sum(trade.get('pnl', 0) for trade in self.position_history)
        except Exception as e:
            logger.error(f"Erreur calcul PnL: {e}")
            return None

    def _calculate_supertrend(self, data):
        """Calcule l'indicateur Supertrend"""
        try:
            # Vérifie si toute la configuration nécessaire est présente
            if not (self.config.get("INDICATORS", {}).get("trend", {}).get("supertrend", {})):
                self.dashboard.update_indicator_status("Supertrend", "DISABLED - Missing config")
                return None
            
            # Récupère les paramètres de configuration
            try:
                period = self.config["INDICATORS"]["trend"]["supertrend"]["period"]
                multiplier = self.config["INDICATORS"]["trend"]["supertrend"]["multiplier"]
            except KeyError:
                self.dashboard.update_indicator_status("Supertrend", "DISABLED - Missing parameters")
                return None
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calcul de l'ATR
            tr = pd.DataFrame()
            tr['h-l'] = high - low
            tr['h-pc'] = abs(high - close.shift(1))
            tr['l-pc'] = abs(low - close.shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            atr = tr['tr'].rolling(period).mean()
            
            # Calcul des bandes
            hl2 = (high + low) / 2
            final_upperband = hl2 + (multiplier * atr)
            final_lowerband = hl2 - (multiplier * atr)
            
            # Calcul du Supertrend
            supertrend = pd.Series(index=data.index)
            direction = pd.Series(index=data.index)
            
            for i in range(period, len(data)):
                if close[i] > final_upperband[i-1]:
                    supertrend[i] = final_lowerband[i]
                    direction[i] = 1
                elif close[i] < final_lowerband[i-1]:
                    supertrend[i] = final_upperband[i]
                    direction[i] = -1
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]
            
            # Si on arrive ici, l'indicateur est calculé avec succès
            self.dashboard.update_indicator_status("Supertrend", "ACTIVE")
            
            return {
                "value": supertrend,
                "direction": direction,
                "strength": abs(close - supertrend) / close
            }
        except Exception as e:
            logger.error(f"Erreur: {e}")
            self.dashboard.update_indicator_status("Supertrend", "ERROR - Calculation failed")
            return None

    def initialize_models(self):
        """Initialise les modèles d'IA"""
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
        logger.error(f"Erreur: {e}")
        return None

async def get_real_time_data(self):
    """Récupère les données en temps réel"""
    try:
        latest_data = await self.websocket.get_latest_data()
        orderbook = await self.exchange.get_orderbook(self.trading_pair)
        volume_24h = await self.exchange.get_24h_volume(self.trading_pair)
        account_balance = await self.exchange.get_balance()
        open_positions = await self.exchange.get_open_positions()
        
        return {
            'price_data': latest_data,
            'orderbook': orderbook,
            'volume': volume_24h,
            'balance': account_balance,
            'positions': open_positions
        }
    except Exception as e:
        logger.error(f"Erreur récupération données temps réel: {e}")
        return None

async def update_real_time(self):
    """Met à jour toutes les données en temps réel"""
    while True:
        try:
            real_time_data = await self.get_real_time_data()
            if real_time_data:
                self.latest_data = real_time_data
                
                # Mise à jour du buffer circulaire
                self.buffer.update(real_time_data)
                
                # Notification des changements importants
                await self.check_significant_changes(real_time_data)
                
            await asyncio.sleep(1)  # Mise à jour chaque seconde
        except Exception as e:
            logger.error(f"Erreur mise à jour temps réel: {e}")
            await asyncio.sleep(5)  # Attente plus longue en cas d'erreur

async def initialize(self):
    """Initialise les connexions asynchrones"""
    try:
        await self.exchange.initialize()
        
        self.binance_ws = await AsyncClient.create(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET')
        )
        self.socket_manager = BinanceSocketManager(self.binance_ws)
        
        await self.setup_streams()
        logger.info("✅ Connexions initialisées avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        raise  # Relance l'exception pour permettre une gestion appropriée
    finally:
        # Nettoyage des ressources si nécessaire
        pass
        
async def shutdown(self):
    """Ferme proprement les connexions"""
    try:
        await self.binance_ws.close_connection()
        await self.connector.close()
        await self.exchange.close()
        logger.info("✅ Connexions fermées avec succès")
    except Exception as e:
        logger.error(f"Erreur fermeture connexions: {e}")

# Point d'entrée principal
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
        # Configuration de l'event loop
        setup_event_loop()
        
        # Instance du bot
        bot = TradingBotM4()
        
        # Interface Streamlit
        st.title("Trading Bot Ultimate v4 🤖")

        # Colonnes pour l'interface
        col1, col2 = st.columns([3, 1])

        with col2:
            # Informations de session
            st.sidebar.info(f"""
            **Session Info**
            User: {os.getenv('CURRENT_USER', 'Patmoorea')}
            Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
            """)
            
            # Configuration trading
            st.header("Trading Configuration")
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Medium"
            )
            pairs = st.multiselect(
                "Trading Pairs",
                options=config["TRADING"]["pairs"],
                default=config["TRADING"]["pairs"]
            )

        with col1:
            # Zone principale pour les graphiques
            st.header("Market Analysis")
            
            # Bouton de démarrage
            if st.button("Start Trading Bot", type="primary"):
                with st.spinner("Initialisation du bot de trading..."):
                    asyncio.run(bot.run())

    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        if 'st' in globals():
            st.error(f"Une erreur est survenue: {e}")
        logging.error("Fatal error", exc_info=True)
