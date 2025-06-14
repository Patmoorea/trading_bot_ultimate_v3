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
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(connect_ws())
                finally:
                    loop.close()
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
        """Récupère les dernières données de marché en temps réel"""
        try:
            # Structure pour stocker les données
            data = {}
        
            # Vérification de la connexion WebSocket
            if not hasattr(self, 'binance_ws') or self.binance_ws is None:
                logger.warning("🔄 WebSocket non initialisé, tentative d'initialisation...")
                if not self.initialized:
                    await self.initialize()
                return None

            # Mise à jour de la date
            self.current_date = "2025-06-14 17:09:42"
            logger.info(f"📅 Mise à jour données à {self.current_date}")

            # Récupération des données pour chaque paire
            for pair in config["TRADING"]["pairs"]:
                logger.info(f"📊 Récupération données pour {pair}")
                data[pair] = {}
            
                try:
                    async def fetch_async():
                        result = {
                            'orderbook': None,
                            'balance': None,
                            'ticker_24h': None,
                            'ticker': None
                        }
                        
                        # 1. Prix en temps réel via WebSocket
                        if hasattr(self.binance_ws, 'get_symbol_ticker'):
                            result['ticker'] = await self.binance_ws.get_symbol_ticker(symbol=pair.replace('/', ''))
                        
                        # 2. & 3. Orderbook et Balance
                        if hasattr(self, 'spot_client'):
                            result['orderbook'] = await self.spot_client.get_order_book(pair)
                            result['balance'] = await self.spot_client.get_balance()
                            
                        # 4. Volume 24h
                        if hasattr(self.binance_ws, 'get_24h_ticker'):
                            result['ticker_24h'] = await self.binance_ws.get_24h_ticker(pair.replace('/', ''))
                            
                        return result

                    # Execution avec timeout correct
                    async with asyncio.timeout(5.0):
                        result = await fetch_async()
                    
                    # Traitement des résultats
                    if result['ticker']:
                        data[pair]['price'] = float(result['ticker']['price'])
                        data[pair]['timestamp'] = self.current_date
                        logger.info(f"💰 Prix {pair}: {data[pair]['price']}")
                    
                    if result['orderbook']:
                        data[pair]['orderbook'] = {
                            'bids': result['orderbook']['bids'][:5],
                            'asks': result['orderbook']['asks'][:5]
                        }
                        logger.info(f"📚 Orderbook mis à jour pour {pair}")
                        
                    if result['balance']:
                        data[pair]['account'] = result['balance']
                        logger.info(f"💼 Balance mise à jour: {result['balance'].get('total', 0)} USDC")
                        
                    if result['ticker_24h']:
                        data[pair].update({
                            'volume': float(result['ticker_24h']['volume']),
                            'price_change': float(result['ticker_24h']['priceChangePercent'])
                        })
                        logger.info(f"📈 Volume 24h {pair}: {data[pair]['volume']}")

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
                        # Mise à jour du buffer circulaire
                        self.buffer.update_data(symbol, symbol_data)
                    
                        # Mise à jour des données latest
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

    async def trading_loop(self):
        """Boucle principale de trading"""
        while True:
            try:
                # Mise à jour des données
                data = await self.get_latest_data()
                if data:
                    for pair in config["TRADING"]["pairs"]:
                        # Calcul des indicateurs pour chaque symbole
                        indicators = await self.calculate_indicators(pair)
                        if indicators:
                            # Analyse des signaux
                            signals = await self.analyze_signals(data)
                        
                            if signals:
                                # Construction de la décision
                                decision = await self.analyze_signals(data, indicators)
                            
                                if decision and decision.get('should_trade', False):
                                    trade_result = await self.execute_real_trade(decision)
                                    if trade_result:
                                        logger.info(f"Trade exécuté: {trade_result['id']}")
                                    
                    # Mise à jour du portfolio
                    await self.get_real_portfolio()
                
                # Attente avant la prochaine itération
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Erreur dans la boucle: {str(e)}")
                await asyncio.sleep(5)
                
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
            def trading_loop():
                while st.session_state.bot_running:
                    try:
                        # Création d'un nouveau loop pour le thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                        # Récupération des données
                        market_data = loop.run_until_complete(self.get_latest_data())
                        if market_data:
                            # Calcul des indicateurs
                            indicators = loop.run_until_complete(
                                self.calculate_indicators('BTC/USDC')
                            )
                        
                            # Analyse des signaux
                            decision = loop.run_until_complete(
                                self.analyze_signals(market_data, indicators)
                            )
                        
                            if decision and decision.get('should_trade', False):
                                trade_result = loop.run_until_complete(
                                    self.execute_real_trade(decision)
                                )
                                if trade_result:
                                    logger.info(f"Trade exécuté: {trade_result['id']}")
                        
                            # Mise à jour du portfolio
                            portfolio = loop.run_until_complete(self.get_real_portfolio())
                        
                            # Mise à jour de l'état
                            if portfolio:
                                st.session_state.portfolio = portfolio
                                st.session_state.latest_data = market_data
                                st.session_state.indicators = indicators
                            
                        time.sleep(1)  # Délai entre les itérations
                    
                    except Exception as loop_error:
                        logger.error(f"Erreur dans la boucle: {loop_error}")
                        time.sleep(5)
                        continue
                    finally:
                        loop.close()

            # Démarrage du thread de trading
            trading_thread = threading.Thread(target=trading_loop)
            trading_thread.daemon = True  # Le thread s'arrêtera quand le programme principal s'arrête
        
            # Stockage du thread dans la session
            st.session_state.trading_thread = trading_thread
        
            # Démarrage du thread
            trading_thread.start()
        
            # Mise à jour de l'état du bot
            st.session_state.bot_running = True
        
            # Message de confirmation
            logger.info("✅ Bot de trading démarré avec succès")
                
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
            st.session_state.bot_running = False
        
        # Notification Telegram en cas d'erreur
        if hasattr(self, 'telegram'):
            try:
                await self.telegram.send_message(
                    f"🚨 Erreur critique du bot:\n{str(e)}\n"
                    f"Trader: {self.current_user}"
                )
            except:
                pass
        raise

    async def create_dashboard(self):
        """Crée le dashboard Streamlit"""
        try:
            # Récupération du portfolio
            portfolio = await self.get_real_portfolio()
            if not portfolio:
                st.error("Unable to fetch portfolio data")
                return

            # En-tête
            st.title("Trading Bot Ultimate v4 🤖")
        
            # Info session
            st.info(f"""
            **Session Info**
            👤 User: {bot.current_user}
            📅 Date: {bot.current_date}
            🚦 Status: {'🟢 Trading' if st.session_state.bot_running else '🔴 Stopped'}
            """)

            # Tabs pour organiser l'information
            tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Trading", "Analysis", "Settings"])

            # TAB 1: PORTFOLIO
            with tab1:
                # Métriques principales sur 4 colonnes
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Value",
                        f"${portfolio['total_value']:,.2f}",
                        delta=f"{portfolio.get('daily_pnl', 0):+.2f}%"
                    )
                with col2:
                    st.metric(
                        "Available USDC",
                        f"${portfolio['free']:,.2f}"
                    )
                with col3:
                    st.metric(
                        "Locked USDC",
                        f"${portfolio['used']:,.2f}"
                    )
                with col4:
                    st.metric(
                        "Available Margin",
                        f"${portfolio['available_margin']:,.2f}"
                    )

                # Positions actuelles
                st.subheader("📊 Active Positions")
                positions_df = pd.DataFrame(portfolio['positions'])
                if not positions_df.empty:
                    st.dataframe(positions_df, use_container_width=True)

            # TAB 2: TRADING
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    # Signaux de trading actifs
                    st.subheader("🎯 Trading Signals")
                    if self.indicators:
                        st.dataframe(pd.DataFrame(self.indicators), use_container_width=True)
            
                with col2:
                    # Ordres en cours
                    st.subheader("📋 Open Orders")
                    if hasattr(self, 'spot_client'):
                        orders = self.spot_client.get_open_orders('BTCUSDC')
                        if orders:
                            st.dataframe(pd.DataFrame(orders), use_container_width=True)

            # TAB 3: ANALYSIS
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    # Indicateurs techniques
                    st.subheader("📉 Technical Analysis")
                    if hasattr(self, 'advanced_indicators'):
                        st.dataframe(
                            pd.DataFrame(self.advanced_indicators.get_all_signals()),
                            use_container_width=True
                        )

            # TAB 4: SETTINGS
            with tab4:
                st.subheader("⚙️ Bot Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    # Paramètres de trading
                    st.write("Trading Parameters")
                    risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 2.0)
                    max_positions = st.number_input("Max Open Positions", 1, 10, 3)
            
            # Sidebar avec contrôles rapides
            with st.sidebar:
                st.header("Quick Controls")
                if st.button("🟢 Start Bot"):
                    await self.run()
                if st.button("🔴 Stop Bot"):
                    await self._cleanup()
            
                st.divider()
            
                # Market Overview
                st.subheader("Market Overview")
                latest_data = self.buffer.get_latest_data() if hasattr(self, 'buffer') else None
                if latest_data:
                    st.metric("BTC/USDC", f"${latest_data.get('price', 0):,.2f}",
                            f"{latest_data.get('change', 0):+.2f}%")

        except Exception as e:
            self.logger.error(f"Erreur création dashboard: {e}")
            st.error(f"Error creating dashboard: {str(e)}")
        
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
            return {"strength": 0, "signals": {}}

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
            return {"current": float('inf'), "threshold": 0.8, "indicators": {}}

    def _analyze_volume_profile(self):
        """Analyse du profil de volume"""
        try:
            volume_data = self.buffer.get_volume_profile()
            if not volume_data:
                return {"supports_entry": False}

            # Calcul des niveaux de support/résistance basés sur le volume
            poc_level = self._calculate_poc(volume_data)
            value_area = self._calculate_value_area(volume_data)
            
            # Analyse de la distribution du volume
            volume_distribution = {
                "above_poc": sum(v for p, v in volume_data.items() if p > poc_level),
                "below_poc": sum(v for p, v in volume_data.items() if p < poc_level)
            }

            # Calcul du ratio de support du volume
            current_price = self.buffer.get_latest_price()
            volume_support = (
                volume_distribution["above_poc"] /
                (volume_distribution["above_poc"] + volume_distribution["below_poc"])
                if current_price > poc_level
                else volume_distribution["below_poc"] /
                (volume_distribution["above_poc"] + volume_distribution["below_poc"])
            )

            return {
                "supports_entry": volume_support > 0.6,
                "poc": poc_level,
                "value_area": value_area,
                "distribution": volume_distribution
            }

        except Exception as e:
            logger.error(f"Erreur: {e}")
            return {"supports_entry": False}

    def _calculate_poc(self, volume_profile):
        """Calcul du Point of Control"""
        try:
            if not volume_profile:
                return None
            return max(volume_profile.items(), key=lambda x: x[1])[0]
        except Exception as e:
            logger.error(f"Erreur calcul POC: {e}")
            return None

    def _calculate_value_area(self, volume_profile, value_area_pct=0.68):
        """Calcul de la Value Area"""
        try:
            if not volume_profile:
                return None

            # Trier les prix par volume décroissant
            sorted_prices = sorted(
                volume_profile.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Calculer le volume total
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * value_area_pct
            cumulative_volume = 0
            value_area_prices = []

            # Construire la value area
            for price, volume in sorted_prices:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= target_volume:
                    break

            return {
                "high": max(value_area_prices),
                "low": min(value_area_prices)
            }

        except Exception as e:
            logger.error(f"Erreur calcul Value Area: {e}")
            return None

    async def run(self):
        """Point d'entrée principal du bot"""
        try:
            logger.info(f"🚀 Démarrage du bot - {self.current_date}")
            logger.info(f"👤 Trader: {self.current_user}")

            # Configuration initiale
            await self.setup_streams()
            
            # Étude initiale du marché
            market_regime, historical_data, initial_analysis = await self.study_market()
            
            while True:
                try:
                    # Mise à jour des données
                    market_data = await self.get_latest_data()
                    if not market_data:
                        continue

                    # Analyse technique
                    signals = await self.analyze_signals(market_data)
                    
                    # Analyse des news
                    news_impact = await self.news_analyzer.analyze()
                    
                    # Construction des features
                    features = self._combine_features(
                        technical_features=signals,
                        news_impact=news_impact,
                        regime=market_regime
                    )
                    
                    # Obtention de la politique et valeur
                    policy, value = self.decision_model(features)
                    
                    if policy is not None and value is not None:
                        # Construction de la décision
                        decision = self._build_decision(
                            policy=policy,
                            value=value,
                            technical_score=signals["recommendation"]["confidence"],
                            news_sentiment=news_impact,
                            regime=market_regime,
                            timestamp=pd.Timestamp.utcnow()
                        )
                        
                        # Ajout gestion des risques
                        decision = self._add_risk_management(decision)
                        
                        # Exécution des trades
                        await self.execute_trades(decision)
                    
                    # Attente avant la prochaine itération
                    await asyncio.sleep(config["TRADING"]["update_interval"])
                    
                except Exception as loop_error:
                    logger.error(f"Erreur dans la boucle principale: {loop_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Erreur fatale: {e}")
            await self.telegram.send_message(
                f"🚨 Erreur critique du bot:\n{str(e)}\n"
                f"Trader: {self.current_user}"
            )
            raise

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
                    asyncio.run(bot.async_init())
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
        
async def update_dashboard(self):
    """Met à jour le dashboard en temps réel"""
    try:
        # Récupération du portfolio
        portfolio = await self.get_real_portfolio()
        
        # Mise à jour des métriques
        st.session_state.portfolio = portfolio
        st.session_state.latest_data = self.latest_data
        st.session_state.indicators = self.indicators
        
        # Force la mise à jour
        st.rerun()
        
    except Exception as e:
        logger.error(f"Erreur mise à jour dashboard: {e}")
        
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
                    
def main():
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
                            def trading_loop():
                                running = True  # Variable locale pour contrôler le thread
                                while running:  # Utiliser la variable locale au lieu de st.session_state
                                    try:
                                        logger.info("🔄 Trading loop iteration started")  # Nouveau log
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
            
                                        market_data = loop.run_until_complete(bot.get_latest_data())
                                        if market_data:
                                            indicators = loop.run_until_complete(
                                                bot.calculate_indicators('BTC/USDC')
                                            )
                                            logger.info(f"📈 Indicators calculated: {indicators}")
                                            portfolio = loop.run_until_complete(bot.get_real_portfolio())
                
                                            # Mise à jour directe sur l'objet bot plutôt que session_state
                                            bot.latest_data = market_data
                                            bot.indicators = indicators
                                            bot.portfolio = portfolio
                
                                        time.sleep(1)
            
                                    except Exception as loop_error:
                                        logger.error(f"Loop error: {loop_error}")
                                        time.sleep(5)
                                    finally:
                                        loop.close()
                
                            # Démarrage du thread de trading
                            trading_thread = threading.Thread(target=trading_loop)
                            trading_thread.daemon = True
                            st.session_state.trading_thread = trading_thread
                            st.session_state.bot_running = True
                            trading_thread.start()
                
                            st.success("✅ Bot is now trading!")
                
                    except Exception as e:
                        st.error(f"❌ Failed to start bot: {str(e)}")
                        logger.error(f"Start error: {e}")
            else:
                if st.button("🔴 Stop Trading", use_container_width=True):
                    try:
                        with st.spinner("Stopping trading bot..."):
                            bot.running = False  # Utilisez une propriété du bot
                            if 'trading_thread' in st.session_state:
                                st.session_state.trading_thread.join(timeout=5)
                        asyncio.run(bot._cleanup())
                        st.session_state.bot_running = False
                        st.success("✅ Bot stopped successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to stop bot: {str(e)}")

            # Status indicator
            st.markdown("---")
            st.markdown(f"**Bot Status**: {'🟢 Running' if st.session_state.bot_running else '🔴 Stopped'}")

            # Main Content - Using tabs
            tabs = st.tabs(["📈 Portfolio", "🎯 Trading", "📊 Analysis"])

            # Portfolio tab
            with tabs[0]:
                if st.session_state.bot_running:
                    # Debug info
                    st.info(f"""
                    **Debug Information**
                    WebSocket: {bot.ws_connection.get('status', 'Unknown')}
                    Last Data Update: {bot.current_date}
                    Data Available: {bool(bot.latest_data)}
                    Indicators Available: {bool(bot.indicators)}
                    """)
                    try:
                        # Utiliser les données en cache plutôt que de refaire un appel
                        portfolio = st.session_state.get('portfolio')
                        if portfolio:
                            # Portfolio Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "💰 Portfolio Value",
                                    f"{portfolio['total_value']:.2f} USDC",
                                    f"{portfolio.get('daily_pnl', 0):+.2f} USDC"
                                )
                            with col2:
                                st.metric(
                                    "📈 24h Volume",
                                    f"{portfolio.get('volume_24h', 0):.2f} USDC",
                                    f"{portfolio.get('volume_change', 0):+.2f}%"
                                )
                            with col3:
                                positions_count = len(portfolio.get('positions', []))
                                st.metric(
                                    "🔄 Active Positions",
                                    str(positions_count),
                                    f"{positions_count} active"
                                )

                         # Active Positions
                            st.subheader("Active Positions")
                            if portfolio.get('positions'):
                                st.dataframe(
                                    pd.DataFrame(portfolio['positions']),
                                    use_container_width=True
                                )
                            else:
                                st.info("💡 No active positions")
                        else:
                            st.warning("⚠️ No portfolio data available")
                    except Exception as e:
                        st.error(f"❌ Error loading portfolio: {str(e)}")
                else:
                    st.warning("⚠️ Bot is not running. Click 'Start Trading' to begin.")

            # Trading tab
            with tabs[1]:
                if st.session_state.bot_running:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "BTC/USDC Price",
                            f"{bot.latest_data.get('price', 0):.2f}",
                            f"{bot.latest_data.get('price_change', 0):+.2f}%"
                        )
                    with col2:
                        st.metric(
                            "Trading Volume",
                            f"{bot.latest_data.get('volume', 0):.2f}",
                             f"{bot.latest_data.get('volume_change', 0):+.2f}%"
                        )
                        
                    # Utiliser les données en cache
                    indicators = st.session_state.get('indicators')
                    if indicators:
                        st.subheader("Trading Signals")
                        st.dataframe(
                            pd.DataFrame(indicators),
                            use_container_width=True
                        )
                    else:
                        st.info("💡 No trading signals available yet")
                else:
                    st.warning("⚠️ Start the bot to see trading signals")

            #  Analysis tab
            with tabs[2]:
                if st.session_state.bot_running:
                    if hasattr(bot, 'advanced_indicators'):
                        try:
                            st.subheader("Technical Analysis")
                            analysis = bot.advanced_indicators.get_all_signals()
                            st.dataframe(pd.DataFrame(analysis), use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Error getting analysis: {str(e)}")
                    else:
                        st.info("💡 No technical analysis available yet")
                else:
                    st.warning("⚠️ Start the bot to see technical analysis")

            # Auto-refresh si le bot est en marche
            if st.session_state.bot_running:
                # Update footer avec timestamp
                st.sidebar.markdown("---")
                st.sidebar.text(f"Updates: {st.session_state.get('refresh_count', 0)}")
                st.sidebar.text(f"Last Update: {bot.current_date}")
    
                # Attendre avant la prochaine mise à jour
                time.sleep(0.5)  # Réduit à 0.5 seconde pour des mises à jour plus fréquentes
                st.rerun()

    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Main error: {e}")

if __name__ == "__main__":
    main()
