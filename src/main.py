import asyncio
import threading
import time
from asyncio import TimeoutError
# 1. Imports système (DOIVENT ÊTRE EN PREMIER)
import os
import sys
import logging
import json
import plotly.graph_objects as go
import re
import time
import threading
from datetime import timezone

# Ajout du chemin racine au PYTHONPATH
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.regime_detection.hmm_kmeans import MarketRegimeDetector
from src.monitoring.streamlit_ui import TradingDashboard
from src.data.realtime.websocket.client import MultiStreamManager, StreamConfig
from src.core.buffer.circular_buffer import CircularBuffer
from src.core.data import CircularBuffer
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
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner as ArbitrageEngine
from src.liquidity_heatmap.visualization import generate_heatmap

# Imports des indicateurs
from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators
from src.analysis.indicators.momentum.momentum import MomentumIndicators
from src.analysis.indicators.volume.volume_analysis import VolumeAnalysis
from src.analysis.indicators.trend.indicators import TrendIndicators

# Dans les imports, ajoutons les composants existants
from src.binance.binance_ws import AsyncClient, BinanceSocketManager
from src.connectors.binance import BinanceConnector
from src.exchanges.binance.binance_client import BinanceClient
from src.news_integration.news_processor import NewsProcessor as NewsAnalyzer

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
load_dotenv()
config = {
    'NEWS': {
        'enabled': True,
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN', '')
    },
    'BINANCE': {
        'API_KEY': os.getenv('BINANCE_API_KEY'),
        'API_SECRET': os.getenv('BINANCE_API_SECRET')
    },
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

@st.cache_resource
def get_bot():
    """Create or get the bot instance"""
    try:
        bot = TradingBotM4()
        bot.current_date = "2025-06-14 00:25:31"
        bot.current_user = "Patmoorea"
        
        # Configuration du WebSocket
        bot.ws_connection = {
            'enabled': False,
            'reconnect_count': 0,
            'max_reconnects': 3,
            'last_connection': None,
            'status': 'disconnected'
        }
        logger.info(f"WebSocket Status: {bot.ws_connection['status']}")
        
        return bot
    except Exception as e:
        # Log l'erreur mais ne pas crasher
        logger.error(f"Error creating bot instance: {e}")
        return None
        
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

def initialize_websocket(bot):
        """Initialize WebSocket connection"""
        try:
            if not bot.ws_connection['enabled']:
                async def connect_ws():
                    bot.binance_ws = await AsyncClient.create(
                        api_key=os.getenv('BINANCE_API_KEY'),
                        api_secret=os.getenv('BINANCE_API_SECRET')
                    )
                    bot.socket_manager = BinanceSocketManager(bot.binance_ws)
                
                    # Démarrer les streams nécessaires
                    streams = []
                
                    # Stream de ticker
                    ticker_socket = bot.socket_manager.symbol_ticker_socket('BTCUSDC')
                    streams.append(ticker_socket)
                
                    # Stream d'orderbook
                    depth_socket = bot.socket_manager.depth_socket('BTCUSDC')
                    streams.append(depth_socket)
                
                    # Stream de klines
                    kline_socket = bot.socket_manager.kline_socket('BTCUSDC', '1m')
                    streams.append(kline_socket)
                
                    # Démarrer tous les streams
                    for stream in streams:
                        asyncio.create_task(handle_socket_message(bot, stream))
                
                    bot.ws_connection['enabled'] = True
                    bot.ws_connection['last_connection'] = time.time()
                
                # Exécuter la connexion WebSocket
                asyncio.run(connect_ws())
                return True
        except Exception as e:
            logger.error(f"WebSocket initialization error: {e}")
            return False
async def handle_socket_message(bot, socket):
    """Handle incoming WebSocket messages"""
    try:
        async with socket as ts:
            while True:
                msg = await ts.recv()
                if msg:
                    await process_ws_message(bot, msg)
    except Exception as e:
        logger.error(f"Socket error: {e}")
        bot.ws_connection['enabled'] = False

async def process_ws_message(bot, msg):
    """Process WebSocket messages"""
    try:
        if msg.get('e') == 'ticker':
            # Mise à jour du prix
            bot.latest_data['price'] = float(msg['c'])
            bot.latest_data['volume'] = float(msg['v'])
            
        elif msg.get('e') == 'depth':
            # Mise à jour de l'orderbook
            bot.latest_data['orderbook'] = {
                'bids': msg['b'][:5],
                'asks': msg['a'][:5]
            }
            
        elif msg.get('e') == 'kline':
            # Mise à jour des klines
            k = msg['k']
            bot.latest_data['klines'] = {
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v'])
            }
    except Exception as e:
        logger.error(f"Message processing error: {e}")
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
    """Classe principale du bot de trading v4 - Version unifiée et mise à jour le 2025-06-10 18:48:29"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialisation du client Binance
        try:
            self.spot_client = BinanceClient(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_API_SECRET')
            )
            logger.info("✅ Spot client initialisé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation spot client: {e}")
            self.spot_client = None
        
        """Initialisation du bot de trading"""
        self.buffer = CircularBuffer(maxlen=1000, compress=True)
        self.indicators = {}
        self.latest_data = {}
        self.config = {
            'NEWS': {
                'enabled': True,
                'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN', '')
            },
            'BINANCE': {
                'API_KEY': os.getenv('BINANCE_API_KEY', ''),
                'API_SECRET': os.getenv('BINANCE_API_SECRET', '')
            }
        }
        self.spot_client = None
        self.ws_manager = None
        
        # Configuration utilisateur et date
        self.current_date = "2025-06-10 18:48:29"
        self.current_user = "Patmoorea"
        self.news_analyzer = None
        self.initialized = False
        
        # Mode de trading
        self.trading_mode = os.getenv('TRADING_MODE', 'production')
        self.testnet = False

        # Activation des composants
        self.news_enabled = True
        self.arbitrage_enabled = True
        self.telegram_enabled = True

        # Configuration risque
        self.max_drawdown = 0.05  # 5% max
        self.daily_stop_loss = 0.02  # 2% par jour
        self.max_position_size = 1000  # USDC

        # Configuration des streams
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

         # Configuration Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
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

        # Configuration timeframes
        self.timeframe_config = TimeframeConfig(
            timeframes=config["TRADING"]["timeframes"],
            weights={
                "1m": 0.1, "5m": 0.15, "15m": 0.2,
                "1h": 0.25, "4h": 0.15, "1d": 0.15
            }
        )
    async def _initialize_models(self):
        """Initialise les modèles d'IA"""
        try:
            # Calcul des dimensions pour CNNLSTM
            input_shape = (
                len(config["TRADING"]["timeframes"]),  # Nombre de timeframes
                len(config["TRADING"]["pairs"]),       # Nombre de paires
                42                                     # Nombre de features par candlestick
            )
        
            # Calcul des dimensions pour PPO-GTrXL
            state_dim = input_shape[0] * input_shape[1] * input_shape[2]
            action_dim = len(config["TRADING"]["pairs"])
        
            # Initialisation des modèles
            self.models = {
                "ppo_gtrxl": PPOGTrXL(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    num_layers=config["AI"]["gtrxl_layers"],
                    d_model=config["AI"]["embedding_dim"]
                ),
                "cnn_lstm": CNNLSTM(input_shape=input_shape)
            }
        
            # Chargement des poids pré-entraînés
            models_path = os.path.join(current_dir, "models")
            if os.path.exists(models_path):
                for model_name, model in self.models.items():
                    model_path = os.path.join(models_path, f"{model_name}.pt")
                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path))
                        logger.info(f"Modèle {model_name} chargé avec succès")
                    
            logger.info("✅ Modèles initialisés avec succès")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur initialisation modèles: {e}")
            return False

    async def _cleanup(self):
        """Nettoie les ressources avant de fermer"""
        try:
            # Fermeture des connexions WebSocket
            if hasattr(self, 'socket_manager'):
                await self.socket_manager.close()
            
            # Fermeture de la connexion Binance
            if hasattr(self, 'binance_ws'):
                await self.binance_ws.close_connection()
            
            # Sauvegarde des modèles
            if hasattr(self, 'models'):
                models_path = os.path.join(current_dir, "models")
                os.makedirs(models_path, exist_ok=True)
                for model_name, model in self.models.items():
                    model_path = os.path.join(models_path, f"{model_name}.pt")
                    torch.save(model.state_dict(), model_path)
                
            # Sauvegarde des métriques
            if hasattr(self, 'dashboard'):
                await self.dashboard.save_metrics()
            
                logger.info("✅ Nettoyage effectué avec succès")
        
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {e}")

    def check_ws_connection(bot):
        """Check WebSocket connection and reconnect if needed"""
        try:
            if not bot.ws_connection['enabled']:
                if bot.ws_connection['reconnect_count'] < bot.ws_connection['max_reconnects']:
                    logger.info("Attempting WebSocket reconnection...")
                    if initialize_websocket(bot):
                        bot.ws_connection['reconnect_count'] = 0
                        return True
                    bot.ws_connection['reconnect_count'] += 1
                else:
                    logger.error("Max WebSocket reconnection attempts reached")
                    return False
            return True
        except Exception as e:
            logger.error(f"WebSocket check error: {e}")
            return False
    
    async def initialize(self):
        """Initialisation asynchrone des connexions"""
        if not self.initialized:
            try:
                # Configuration Binance
                self.binance_ws = await AsyncClient.create(
                    api_key=os.getenv('BINANCE_API_KEY'),
                    api_secret=os.getenv('BINANCE_API_SECRET')
                )
                self.socket_manager = BinanceSocketManager(self.binance_ws)
                
                # Client Binance standard (non async)
                self.spot_client = BinanceClient(
                    api_key=os.getenv('BINANCE_API_KEY'),
                    api_secret=os.getenv('BINANCE_API_SECRET')
                )
                
                # Configuration de l'exchange ccxt pour le portfolio
                self.exchange = ccxt.binance({
                    'apiKey': os.getenv('BINANCE_API_KEY'),
                    'secret': os.getenv('BINANCE_API_SECRET'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True
                    }
                })
                
                # Configuration des streams
                self.stream_config = StreamConfig(
                    max_connections=12,
                    reconnect_delay=1.0,
                    buffer_size=10000
                )

                # Initialisation des composants
                await self._setup_components()
                
                # Test de récupération du portfolio
                portfolio = await self.get_real_portfolio()
                if not portfolio:
                    logger.warning("Unable to fetch initial portfolio data")
                
                self.initialized = True
                logger.info("Bot initialized successfully")
            except Exception as e:
                self.logger.error(f"Initialization error: {e}")
                raise
            
    async def _setup_components(self):
        """Configure les composants du bot"""
        try:
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
            
            # News Analyzer
            self.news_analyzer = NewsAnalyzer()
            
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
            
            # Configuration des analyseurs et modèles
            await self._initialize_analyzers()
            await self._initialize_models()
            
        except Exception as e:
            logger.error(f"Setup components error: {e}")
            raise

    async def _initialize_analyzers(self):
        """Initialize all analysis components"""
        self.advanced_indicators = MultiTimeframeAnalyzer(
            config=self.timeframe_config
        )
        self.orderflow_analysis = OrderFlowAnalysis(
            config=OrderFlowConfig(tick_size=0.1)
        )
        self.volume_analysis = VolumeAnalysis()
        self.volatility_indicators = VolatilityIndicators()

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

    def decision_model(self, features, timestamp=None):
        try:
            policy = self.models["ppo_gtrxl"].get_policy(features)
            value = self.models["ppo_gtrxl"].get_value(features)
            return policy, value
        except Exception as e:
            logger.error(f"[{timestamp}] Erreur decision_model: {e}")
            return None, None

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
        try:
            data = {}
            
            if not self.ws_connection["enabled"]:
                await self.initialize()
                if not self.ws_connection["enabled"]:
                    return None
            
            async def fetch_market_data():
                tasks = []
                for pair in config["TRADING"]["pairs"]:
                    tasks.append(asyncio.create_task(self._fetch_pair_data(pair)))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return {
                    pair: result for pair, result in zip(config["TRADING"]["pairs"], results)
                    if not isinstance(result, Exception)
                }
            
            try:
                market_data = await asyncio.wait_for(fetch_market_data(), timeout=5.0)
                if market_data:
                    for pair, ticker_data in market_data.items():
                        self.buffer.update_data(pair, ticker_data)
                        self.latest_data[pair] = ticker_data
                    return market_data
                return None
                
            except asyncio.TimeoutError:
                logger.warning("⏱️ Timeout récupération données")
                return None

        except Exception as e:
            logger.error(f"❌ Erreur critique: {str(e)}")
            return None

            # Récupération des données pour chaque paire
            for pair in config["TRADING"]["pairs"]:
                logger.info(f"📊 Récupération données pour {pair}")
                data[pair] = {}
            
                try:
                    # Création des tâches asynchrones pour chaque type de données
                    async def fetch_ticker():
                        if hasattr(self.binance_ws, 'get_symbol_ticker'):
                            ticker = await self.binance_ws.get_symbol_ticker(symbol=pair.replace('/', ''))
                            if ticker:
                                return float(ticker['price'])
                        return None

                    async def fetch_orderbook():
                        if hasattr(self, 'spot_client'):
                            orderbook = await self.spot_client.get_order_book(pair)
                            if orderbook:
                                return {
                                    'bids': orderbook['bids'][:5],
                                    'asks': orderbook['asks'][:5]
                                }
                        return None

                    async def fetch_balance():
                        if hasattr(self, 'spot_client'):
                            return await self.spot_client.get_balance()
                        return None

                    async def fetch_24h_ticker():
                        if hasattr(self.binance_ws, 'get_24h_ticker'):
                            ticker_24h = await self.binance_ws.get_24h_ticker(pair.replace('/', ''))
                            if ticker_24h:
                                return {
                                    'volume': float(ticker_24h['volume']),
                                    'price_change': float(ticker_24h['priceChangePercent'])
                                }
                        return None

                    # Exécution des tâches avec timeout
                    tasks = [
                        asyncio.create_task(fetch_ticker()),
                        asyncio.create_task(fetch_orderbook()),
                        asyncio.create_task(fetch_balance()),
                        asyncio.create_task(fetch_24h_ticker())
                    ]

                    # Attendre les résultats avec timeout
                    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
                
                    # Traitement des résultats
                    price, orderbook, balance, ticker_24h = results
                
                    if price is not None:
                        data[pair]['price'] = price
                        logger.info(f"💰 Prix {pair}: {price}")
                
                    if orderbook is not None:
                        data[pair]['orderbook'] = orderbook
                        logger.info(f"📚 Orderbook mis à jour pour {pair}")
                
                    if balance is not None:
                        data[pair]['account'] = balance
                        logger.info(f"💼 Balance mise à jour: {balance.get('total', 0)} USDC")
                
                    if ticker_24h is not None:
                        data[pair].update(ticker_24h)
                        logger.info(f"📈 Volume 24h {pair}: {ticker_24h.get('volume', 0)}")

                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ Timeout pour {pair}")
                    continue
                except Exception as inner_e:
                    logger.error(f"❌ Erreur récupération données {pair}: {inner_e}")
                    continue

            # Mise en cache des données si disponibles
            if data and any(data.values()):
                logger.info("✅ Données reçues, mise à jour du buffer")
                for symbol, symbol_data in data.items():
                    if symbol_data:
                        self.buffer.update_data(symbol, symbol_data)
                        self.latest_data[symbol] = symbol_data
                return data
            else:
                logger.warning("⚠️ Aucune donnée reçue")
                return None

        except Exception as e:
            logger.error(f"❌ Erreur critique get_latest_data: {e}")
            return None

    async def calculate_indicators(self, symbol: str) -> dict:
        """Calcule les indicateurs techniques"""
        try:
            data = self.latest_data.get(symbol)
            if not data:
                logger.error(f"❌ Pas de données pour {symbol}")
                return {}
            
            # Calcul des indicateurs de base
            indicators = {
                'price': data['price'],
                'volume': data['volume'],
                'bid_ask_spread': data['ask'] - data['bid'],
                'high_low_range': data['high'] - data['low'],
                'timestamp': data['timestamp']
            }
            # Log des données reçues
            logger.info(f"Calcul indicateurs pour {symbol}: {data}")# Log des données reçues
            logger.info(f"Calcul indicateurs pour {symbol}: {data}")
                
            # Stockage des indicateurs
            self.indicators[symbol] = indicators
            return indicators
        
        except Exception as e:
            logger.error(f"Erreur calcul indicateurs pour {symbol}: {str(e)}")
            return {}

                
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

    async def analyze_signals(self, market_data, indicators=None):
        """Analyse des signaux de trading basée sur tous les indicateurs"""
        try:
            # Si les indicateurs ne sont pas fournis, on les calcule
            if indicators is None:
                indicators = self.add_indicators(market_data)
            
            if not indicators:
                return None
        
            # Analyse des tendances
            trend_analysis = {
                'primary_trend': 'bullish' if indicators['trend']['ema_fast'].iloc[-1] > indicators['trend']['sma_slow'].iloc[-1] else 'bearish',
                'trend_strength': indicators['trend']['adx'].iloc[-1],
                'trend_direction': 1 if indicators['trend']['vortex_ind_diff'].iloc[-1] > 0 else -1,
                'ichimoku_signal': 'buy' if indicators['trend']['ichimoku_a'].iloc[-1] > indicators['trend']['ichimoku_b'].iloc[-1] else 'sell'
            }
    
            # Analyse du momentum
            momentum_analysis = {
                'rsi_signal': 'oversold' if indicators['momentum']['rsi'].iloc[-1] < 30 else 'overbought' if indicators['momentum']['rsi'].iloc[-1] > 70 else 'neutral',
                'stoch_signal': 'buy' if indicators['momentum']['stoch_rsi_k'].iloc[-1] > indicators['momentum']['stoch_rsi_d'].iloc[-1] else 'sell',
                'ultimate_signal': 'buy' if indicators['momentum']['uo'].iloc[-1] > 70 else 'sell' if indicators['momentum']['uo'].iloc[-1] < 30 else 'neutral'
            }
    
            # Analyse de la volatilité
            volatility_analysis = {
                'bb_signal': 'oversold' if market_data['close'].iloc[-1] < indicators['volatility']['bbl'].iloc[-1] else 'overbought',
                'kc_signal': 'breakout' if market_data['close'].iloc[-1] > indicators['volatility']['kch'].iloc[-1] else 'breakdown',
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
    
    async def setup_real_exchange(self):
        """Configuration sécurisée de l'exchange"""
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
        
            if not api_key or not api_secret:
                raise ValueError("Clés API Binance manquantes dans les variables d'environnement")
            
            # Configuration de l'exchange avec ccxt
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'createMarketBuyOrderRequiresPrice': False
                }
            })
        
            # Chargement des marchés de manière synchrone
            self.exchange.load_markets()
            self.spot_client = self.exchange
            self.spot_client = BinanceClient(
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_API_SECRET')
            )
            # Test de la connexion
            balance = self.exchange.fetch_balance()
            if not balance:
                raise ValueError("Impossible de récupérer le solde - Vérifiez vos clés API")
            
            logger.info("Exchange configuré avec succès")
            return True
        
        except Exception as e:
            logger.error(f"Erreur configuration exchange: {e}")
            return False
        
    # 3. Correction de l'envoi des messages Telegram
    async def send_telegram_message(self, message: str):
        """Envoie un message via Telegram"""
        try:
            if hasattr(self, 'telegram') and self.telegram.enabled:
                success = await self.telegram.send_message(
                    message=message,
                    parse_mode='HTML'
                )
                if success:
                    logger.info(f"Message Telegram envoyé: {message[:50]}...")
                else:
                    logger.error("Échec envoi message Telegram")
        except Exception as e:
            logger.error(f"Erreur envoi Telegram: {e}")

    async def setup_real_telegram(self):
        """Configuration sécurisée de Telegram"""
        try:
            # Création de l'instance TelegramBot (l'initialisation se fait dans __init__)
            self.telegram = TelegramBot()
        
            if not self.telegram.enabled:
                logger.warning("Telegram notifications désactivées")
                return False
            
            # Démarrage du processeur de queue
            await self.telegram.start()
        
            # Test d'envoi d'un message
            success = await self.telegram.send_message(
                "🤖 Bot de trading démarré",
                parse_mode='HTML'
            )
        
            if success:
                logger.info("Telegram configuré avec succès")
                return True
            else:
                logger.error("Échec du test d'envoi Telegram")
                return False
            
        except Exception as e:
            logger.error(f"Erreur configuration Telegram: {e}")
            return False

    async def get_real_portfolio(self):
        """
        Récupère le portfolio en temps réel avec les balances et positions.
        """
        try:
            if not hasattr(self, 'spot_client') or self.spot_client is None:
                logger.error("❌ Spot client non initialisé")
                 # Log de debug
                logger.info("Récupération du portfolio...")
        
                # Récupération de la balance
                balance = self.spot_client.get_balance()
                logger.info(f"Balance reçue: {balance}")
                
                # Tentative de réinitialisation du spot client
                self.spot_client = BinanceClient(
                    api_key=os.getenv('BINANCE_API_KEY'),
                    api_secret=os.getenv('BINANCE_API_SECRET')
                )
                if not self.spot_client:
                    raise Exception("Impossible d'initialiser le spot client")

            # Récupération de la balance
            balance = self.spot_client.get_balance()
            if not balance or 'balances' not in balance:
                raise Exception("Balance non disponible ou vide")

            self.logger.info("💰 Balance reçue")

            # Extraction des USDC
            usdc_balance = None
            for asset_balance in balance['balances']:
                if asset_balance['asset'] == 'USDC':
                    usdc_balance = {
                        'free': float(asset_balance['free']),
                        'locked': float(asset_balance['locked'])
                    }
                    break

            if not usdc_balance:
                # Si pas d'USDC, on utilise des valeurs par défaut pour le test
                usdc_balance = {
                    'free': 100.59,
                    'locked': 0.0
                }

            total_usdc = usdc_balance['free'] + usdc_balance['locked']

            # Construction du portfolio
            portfolio = {
                'total_value': total_usdc,
                'free': usdc_balance['free'],
                'used': usdc_balance['locked'],
                'positions': [],
                'timestamp': self.current_date,
                'update_user': self.current_user,
                'daily_pnl': 0.0,
                'volume_24h': 0.0,
                'volume_change': 0.0
            }

            # Récupération des positions réelles
            try:
                open_orders = self.spot_client.get_open_orders('BTC/USDC')
                if open_orders:
                    positions = []
                    for order in open_orders:
                        if float(order['amount']) > 0:
                            positions.append({
                                'symbol': order['symbol'],
                                'size': float(order['amount']),
                                'value': float(order['price']) * float(order['amount']),
                                'price': float(order['price']),
                                'side': order['side'].upper(),
                                'timestamp': portfolio['timestamp']
                            })
                    portfolio['positions'] = positions

                self.logger.info(f"📊 {len(portfolio.get('positions', []))} positions réelles récupérées")

            except Exception as e:
                self.logger.warning(f"⚠️ Impossible de récupérer les positions: {e}")

            # Mise à jour des métriques
            portfolio.update({
                'position_count': len(portfolio['positions']),
                'total_position_value': sum(pos['value'] for pos in portfolio['positions']),
                'available_margin': portfolio['free'] - sum(pos['value'] for pos in portfolio['positions'])
            })

            self.logger.info(f"✅ Portfolio mis à jour avec succès: {portfolio['total_value']:.2f} USDC")
            return portfolio

        except Exception as e:
            self.logger.error(f"❌ Erreur critique portfolio: {e}")
            # Retourner un portfolio par défaut en cas d'erreur
            return {
                'total_value': 100.59,
                'free': 100.59,
                'used': 0.0,
                'positions': [],
                'timestamp': self.current_date,
                'update_user': self.current_user,
                'daily_pnl': 0.0,
                'volume_24h': 0.0,
                'volume_change': 0.0
            }

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

            try:
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
            except Exception as msg_error:
                logger.error(f"Erreur envoi notification trade: {msg_error}")

            return order
            
        except Exception as e:
            logger.error(f"Erreur trade: {e}")
            return None

    async def run_real_trading(self):
        """Boucle de trading réel sécurisée"""
        try:
            # Configuration initiale
            if not await self.setup_real_exchange():
                raise Exception("Échec configuration exchange")
        
            if not await self.setup_real_telegram():
                raise Exception("Échec configuration Telegram")

            # Mise à jour de la date et de l'utilisateur
            self.current_date = "2025-06-14 05:11:03"
            self.current_user = "Patmoorea"
        
            logger.info(f"""
╔═════════════════════════════════════════════════════════════╗
║                Trading Bot Ultimate v4 - REAL               ║
╠═════════════════════════════════════════════════════════════╣
║ User: {self.current_user}                                  ║
║ Mode: REAL TRADING                                         ║
║ Status: RUNNING                                            ║
╚═════════════════════════════════════════════════════════════╝
                """)

            # Création d'un thread pour la boucle de trading
    st.title("Trading Bot Ultimate v4 🤖")
    
   # Initialisation de l'état
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = None
    if 'latest_data' not in st.session_state:
        st.session_state.latest_data = None
    if 'indicators' not in st.session_state:
        st.session_state.indicators = None
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    if 'trading_thread' not in st.session_state:
        st.session_state.trading_thread = None
        
    try:
        # Get or create bot instance
        bot = get_bot()
        if bot is None:
            st.error("❌ Failed to initialize bot")
            return

        # Initialize WebSocket if not already done
        if not bot.ws_connection['enabled']:
            with st.spinner("Connecting to WebSocket..."):
                if not initialize_websocket(bot):
                    st.error("❌ Failed to establish WebSocket connection")
                    return
                st.success("✅ WebSocket connected!")

        # Colonne d'état
        status_col1, status_col2 = st.columns([2, 1])
        
        with status_col1:
            st.info(f"""
            **Session Info**
            👤 User: {bot.current_user}
            📅 Date: 2025-06-14 00:24:20 UTC
            🚦 Status: {'🟢 Trading' if st.session_state.bot_running else '🔴 Stopped'}
            """)

        # Sidebar Configuration
        with st.sidebar:
            st.header("🛠️ Bot Controls")
            
            # Risk Level
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Low"
            )
            
            st.divider()
            
            # Control Buttons dans une seule colonne
            if not st.session_state.bot_running:
                if st.button("🟢 Start Trading", use_container_width=True):
                    try:
                        with st.spinner("Starting trading bot..."):
                            # Initialisation si pas déjà fait
                            if not bot.initialized:
                                asyncio.run(bot.initialize())

                            # Définition de la variable running pour le thread
                            running = True
                
                            # Démarrage du trading dans un thread séparé
