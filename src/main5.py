# 1. Import et configuration Streamlit (DOIT ÊTRE EN PREMIER)
import streamlit as st
import os

# Charger les variables d'env
from dotenv import load_dotenv

load_dotenv()

print("BINANCE_TESTNET (os.environ):", os.environ.get("BINANCE_TESTNET"))
print("BINANCE_TESTNET (os.getenv):", os.getenv("BINANCE_TESTNET"))

USE_TESTNET = str(os.getenv("BINANCE_TESTNET", "False")).lower() in ("true", "1")


# --- Ajout: Hack JavaScript pour autorefresh sans st_autorefresh ---
def auto_refresh(interval_ms=2000, key="js_autorefresh"):
    """Inject JS code for auto-refresh in Streamlit."""
    js_code = f"""
    <script>
        if (!window.{key}) {{
            window.{key} = setInterval(function() {{
                window.location.reload();
            }}, {interval_ms});
        }}
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


# Initialisation des flags de protection
for flag, default in [
    ("prevent_cleanup", True),
    ("keep_alive", True),
    ("force_cleanup", False),
    ("cleanup_allowed", False),
]:
    if flag not in st.session_state:
        st.session_state[flag] = default

st.set_page_config(
    page_title="Trading Bot Ultimate v4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Imports système
import sys
import logging
import json
import re
import time
import signal
from datetime import timedelta
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import AsyncExitStack
from asyncio import TimeoutError, AbstractEventLoop
import asyncio
import nest_asyncio
import aiohttp
import traceback

# 3. Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 4. Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 5. Imports des bibliothèques externes
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
import telegram
import ccxt
import ccxt.async_support as ccxt
import ta
import gymnasium as gym
from gymnasium import spaces

# 6. Imports des modules internes (exchanges, core, etc.)
from binance import AsyncClient, BinanceSocketManager
from src.exchanges.binance_exchange import BinanceExchange
from src.exchanges.binance.binance_client import BinanceClient
from src.core.exchange import ExchangeInterface as Exchange
from src.core.buffer.circular_buffer import CircularBuffer
from src.connectors.binance import BinanceConnector
from src.portfolio.real_portfolio import RealPortfolio
from src.regime_detection.hmm_kmeans import MarketRegimeDetector
from src.monitoring.streamlit_ui import TradingDashboard
from src.data.realtime.websocket.client import MultiStreamManager, StreamConfig
from src.indicators.advanced.multi_timeframe import (
    MultiTimeframeAnalyzer,
    TimeframeConfig,
)
from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators
from src.analysis.indicators.momentum.momentum import MomentumIndicators
from src.analysis.indicators.volume.volume_analysis import VolumeAnalysis
from src.analysis.indicators.trend.indicators import TrendIndicators
from src.analysis.indicators.orderflow.orderflow_analysis import (
    OrderFlowAnalysis,
    OrderFlowConfig,
)
from src.analysis.indicators.volatility.volatility import VolatilityIndicators
from src.ai.cnn_lstm import CNNLSTM
from src.ai.ppo_gtrxl import PPOGTrXL
from src.ai.hybrid_model import HybridAI
from src.quantum.qsvm import QuantumTradingModel as QuantumSVM
from src.risk_management.circuit_breakers import CircuitBreaker
from src.risk_management.position_manager import PositionManager
from src.notifications.telegram_bot import TelegramBot
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import (
    ArbitrageScanner as ArbitrageEngine,
)
from src.liquidity_heatmap.visualization import generate_heatmap
from web_interface.app.services.news_analyzer import NewsAnalyzer
from src.backtesting.advanced.quantum_backtest import QuantumBacktester, BacktestConfig
from src.backtesting.core.backtest_engine import BacktestEngine

# 7. Constantes de nettoyage
cleanup_lock = asyncio.Lock()
cleanup_in_progress = False
last_cleanup_time = 0
CLEANUP_COOLDOWN = 5

# 8. Constantes WebSocket globales
WEBSOCKET_CONFIG = {
    "RECONNECT_DELAY": 1.0,
    "MESSAGE_TIMEOUT": 30.0,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 5.0,
    "STREAM_TYPES": ["ticker", "depth", "kline"],
}


def setup_asyncio():
    """Configure l'environnement asyncio pour Streamlit."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply()
        return loop
    except Exception as e:
        logger.error(f"Error setting up asyncio: {e}")
        return None


class StreamlitSessionManager:
    """Gestionnaire de session Streamlit avec protection et logging améliorés"""

    def __init__(self):
        """Initialisation du gestionnaire de session"""
        self.init_time = datetime.now(timezone.utc)
        self.user = os.getenv("USER", "Patmoorea")
        self.session_id = f"{self.user}_{int(self.init_time.timestamp())}"
        self.logger = logging.getLogger(__name__)

        # Initialisation immédiate de la session
        if "session_initialized" not in st.session_state:
            if self._initialize_session_state():
                self._log_initialization()

        # Initialisation de advanced_indicators
        self.advanced_indicators = {}

    # Ajoute cette fonction utilitaire
    def safe_float(self, val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _initialize_session_state(self):
        """Initialise l'état de la session avec des valeurs sûres"""
        try:
            # États par défaut avec horodatage
            default_state = {
                # États de base
                "session_id": self.session_id,
                "initialization_time": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_update_time": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
                "user": self.user,
                "initialized": True,
                "session_initialized": True,
                # États du bot
                "bot_running": False,
                "portfolio": None,
                "latest_data": {},
                "indicators": None,
                "refresh_count": 0,
                # États de la boucle événementielle
                "loop": None,
                "error_count": 0,
                # États WebSocket
                "ws_status": "disconnected",
                "ws_initialized": False,
                "ws_connection_status": "disconnected",
                "ws_last_heartbeat": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
                # Protections
                "keep_alive": True,
                "prevent_cleanup": True,
                "force_cleanup": False,
                "cleanup_allowed": False,
            }

            # Initialisation des états manquants uniquement
            for key, value in default_state.items():
                if key not in st.session_state:
                    st.session_state[key] = value

            return True

        except Exception as e:
            self._log_error("Session state initialization error", e)
            return False

    def _log_initialization(self):
        """Log de l'initialisation de la session"""
        self.logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION INITIALIZED                    ║
╠═════════════════════════════════════════════════╣
║ Time: {self.init_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {self.user}
║ Session ID: {self.session_id}
║ Status: Active
╚═════════════════════════════════════════════════╝
        """
        )

    def _log_error(self, message, error):
        """Log unifié des erreurs"""
        self.logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION ERROR                          ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {message}
║ Details: {str(error)}
║ Type: {type(error).__name__}
║ Session ID: {self.session_id}
╚═════════════════════════════════════════════════╝
        """
        )

        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1

    def _log_protection(self):
        """Log de la protection de session"""
        self.logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION PROTECTED                      ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Session ID: {self.session_id}
║ Last Action: {st.session_state.get('last_action_time')}
╚═════════════════════════════════════════════════╝
        """
        )

    def protect_session(self):
        """Protection renforcée de la session"""
        try:
            # Vérification et réinitialisation si nécessaire
            if not st.session_state.get("session_initialized"):
                self._initialize_session_state()

            # Mise à jour du timestamp
            current_time = datetime.now(timezone.utc)
            st.session_state.last_action_time = current_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Activation des protections
            st.session_state.prevent_cleanup = True
            st.session_state.keep_alive = True
            st.session_state.force_cleanup = False
            st.session_state.cleanup_allowed = False

            self._log_protection()
            return True

        except Exception as e:
            self._log_error("Session protection error", e)
            return False

    def allow_cleanup(self):
        """Autorisation sécurisée du nettoyage"""
        try:
            # Vérification de l'état du bot
            if st.session_state.get("bot_running", False):
                logger.warning("Cannot allow cleanup while bot is running")
                return False

            # Configuration du nettoyage
            st.session_state.cleanup_allowed = True
            st.session_state.force_cleanup = True
            st.session_state.prevent_cleanup = False
            st.session_state.keep_alive = False

            self._log_cleanup_authorization()
            return True

        except Exception as e:
            self._log_error("Cleanup authorization error", e)
            return False

    def get_session_info(self):
        """Récupération des informations de session"""
        try:
            info = {
                "user": self.user,
                "session_id": self.session_id,
                "init_time": self.init_time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_action": st.session_state.get("last_action_time"),
                "session_initialized": st.session_state.get(
                    "session_initialized", False
                ),
                "bot_running": st.session_state.get("bot_running", False),
                "ws_initialized": st.session_state.get("ws_initialized", False),
                "error_count": st.session_state.get("error_count", 0),
            }
            return info

        except Exception as e:
            self._log_error("Session info retrieval error", e)
            return None


def _setup_and_verify_event_loop():
    """Configure et vérifie la boucle d'événements avec gestion d'erreur améliorée"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")

    try:
        # Vérification de l'existence d'une boucle
        if not st.session_state.get("loop"):
            # Création et configuration de la nouvelle boucle
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply()

            # Sauvegarde dans la session
            st.session_state.loop = loop

            # Log de succès d'initialisation
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP INITIALIZED              ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
║ Status: Successfully configured
║ Loop ID: {id(loop)}
╚═════════════════════════════════════════════════╝
            """
            )

            return loop

        # Vérification de la boucle existante
        existing_loop = st.session_state.loop
        if existing_loop.is_closed():
            logger.warning(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP CLOSED                   ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Status: Creating new loop
║ Previous Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
            """
            )

            # Création d'une nouvelle boucle
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            nest_asyncio.apply()
            st.session_state.loop = new_loop
            return new_loop

        # Retour de la boucle existante
        logger.debug(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP VERIFIED                 ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Status: Using existing loop
║ Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
        """
        )

        return existing_loop

    except Exception as e:
        # Log d'erreur détaillé
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
║ Details: {traceback.format_exc()}
╚═════════════════════════════════════════════════╝
        """
        )

        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1

        return None

    finally:
        # Mise à jour du timestamp
        st.session_state.last_update_time = current_time.strftime("%Y-%m-%d %H:%M:%S")


# Création de l'instance globale avec vérification
try:
    session_manager = StreamlitSessionManager()
    logger.info("✅ Session manager initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize session manager: {e}")
    session_manager = None

    # Dans la méthode cleanup()


async def cleanup(self):
    """Nettoie les ressources WebSocket"""
    try:
        self.running = False

        # Annulation des tâches
        for stream in self.streams.values():
            if not stream.done():
                stream.cancel()
                try:
                    await stream
                except asyncio.CancelledError:
                    pass

        self.streams.clear()

        # Fermeture du socket manager
        if hasattr(self.bot, "socket_manager") and self.bot.socket_manager:
            try:
                # Modification pour utiliser stop_socket
                for socket in self.bot.socket_manager.sockets:
                    await self.bot.socket_manager.stop_socket(socket)
                self.bot.socket_manager = None
            except Exception as e:
                self.logger.warning(f"Error closing socket manager: {e}")

        # Fermeture du client Binance
        if hasattr(self.bot, "binance_ws") and self.bot.binance_ws:
            try:
                await self.bot.binance_ws.close_connection()
            except Exception as e:
                self.logger.warning(f"Error closing Binance client: {e}")
            self.bot.binance_ws = None
    except Exception as e:
        self.logger.error(f"Cleanup error: {e}")


# Définition de la classe SessionManager
class SessionManager:
    def __init__(self):
        self.sessions = set()

    def register(self, session):
        self.sessions.add(session)
        logging.getLogger(__name__).info(
            f"New session registered (active: {len(self.sessions)})"
        )

    def unregister(self, session):
        self.sessions.discard(session)
        logging.getLogger(__name__).info(
            f"Session unregistered (remaining: {len(self.sessions)})"
        )


def setup_event_loop() -> AbstractEventLoop:
    """Configure l'event loop pour Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply()
    return loop


def init_session_state():
    """Initialize session state variables with strong defaults"""
    session_vars = {
        "initialized": False,
        "bot_running": False,
        "portfolio": None,
        "latest_data": {},
        "indicators": None,
        "refresh_count": 0,
        "loop": None,
        "ws_status": "disconnected",
        "error_count": 0,
        "keep_alive": True,  # Force à True
        "prevent_cleanup": True,  # Force à True
        "force_cleanup": False,  # Force à False
        "ws_initialized": False,
        "cleanup_allowed": False,  # Nouveau flag
    }

    for var, default in session_vars.items():
        # Ne pas écraser les valeurs existantes pour keep_alive et prevent_cleanup
        if var in ["keep_alive", "prevent_cleanup"]:
            st.session_state.setdefault(var, True)
        else:
            st.session_state[var] = default


# Configuration du bot

config = {
    "NEWS": {"enabled": True, "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", "")},
    "BINANCE": {
        "API_KEY": os.getenv("BINANCE_API_KEY"),
        "API_SECRET": os.getenv("BINANCE_API_SECRET"),
        "TESTNET": USE_TESTNET,
    },
    "ARBITRAGE": {
        "exchanges": ["binance", "bitfinex", "kraken"],
        "min_profit": 0.001,
        "max_trade_size": 1000,
        "pairs": ["BTC/USDC", "ETH/USDC"],
        "timeout": 5,
        "volume_filter": 1000,
        "price_check": True,
        "max_slippage": 0.0005,
    },
    "TRADING": {
        "base_currency": "USDC",
        "pairs": ["BTC/USDC", "ETH/USDC"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "study_period": "7d",
    },
    "RISK": {
        "max_drawdown": 0.05,
        "daily_stop_loss": 0.02,
        "position_sizing": "volatility_based",
        "circuit_breaker": {
            "market_crash": True,
            "liquidity_shock": True,
            "black_swan": True,
        },
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
        "gradient_clip": 0.5,
    },
    "INDICATORS": {
        "trend": {
            "supertrend": {"period": 10, "multiplier": 3},
            "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52},
            "ema_ribbon": [5, 10, 20, 50, 100, 200],
        },
        "momentum": {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "stoch_rsi": {"period": 14, "k": 3, "d": 3},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
        },
        "volatility": {
            "bbands": {"period": 20, "std_dev": 2},
            "keltner": {"period": 20, "atr_mult": 2},
            "atr": {"period": 14},
        },
        "volume": {
            "vwap": {"anchor": "session"},
            "obv": {"signal": 20},
            "volume_profile": {"price_levels": 100},
        },
        "orderflow": {
            "delta": {"window": 100},
            "cvd": {"smoothing": 20},
            "imbalance": {"threshold": 0.2},
        },
    },
}


@st.cache_resource(ttl=None)
def get_bot():
    """Create or get the bot instance with lifecycle protection"""
    if "bot_instance" in st.session_state and st.session_state.bot_instance is not None:
        return st.session_state.bot_instance

    try:
        session_manager.protect_session()  # Protection explicite
        logger.info("Creating new bot instance...")
        bot = TradingBotM4()
        st.session_state.bot_instance = bot
        return bot
    except Exception as e:
        logger.error(f"Bot creation error: {e}")
        return None

        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║             CREATING BOT INSTANCE                ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Création du bot
        bot = TradingBotM4()

        # Configuration de la boucle d'événements
        if not st.session_state.get("loop"):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                nest_asyncio.apply()
                st.session_state.loop = loop
                logger.info("✅ Event loop configured successfully")
            except Exception as loop_error:
                logger.error(
                    f"""
╔═════════════════════════════════════════════════╗
║             EVENT LOOP ERROR                     ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(loop_error)}
╚═════════════════════════════════════════════════╝
                """
                )
                raise

        # Initialisation du bot
        async def initialize_bot():
            try:
                if not await bot.start():
                    raise Exception("Bot initialization failed")
                bot._initialized = True
                logger.info("✅ Bot initialization successful")
                return bot
            except Exception as init_error:
                logger.error(
                    f"""
╔═════════════════════════════════════════════════╗
║             INITIALIZATION ERROR                 ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(init_error)}
╚═════════════════════════════════════════════════╝
                """
                )
                raise

        try:
            # Initialisation avec gestion des erreurs de boucle
            try:
                bot = st.session_state.loop.run_until_complete(initialize_bot())
            except RuntimeError as e:
                if "This event loop is already running" in str(e):
                    logger.warning(
                        "⚠️ Event loop already running, applying nest_asyncio"
                    )
                    nest_asyncio.apply()
                    bot = st.session_state.loop.run_until_complete(initialize_bot())
                else:
                    raise

            if not bot or not getattr(bot, "_initialized", False):
                raise Exception("Bot initialization incomplete")

            # Sauvegarde dans la session state
            st.session_state.bot_instance = bot

            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║             BOT INSTANCE READY                   ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Status: {bot.ws_connection.get('status', 'initializing')}
║ Trading Mode: {getattr(bot, 'trading_mode', 'production')}
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
            """
            )

            return bot

        except Exception as run_error:
            logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║             RUNTIME ERROR                        ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(run_error)}
╚═════════════════════════════════════════════════╝
            """
            )
            # Nettoyage sécurisé
            if hasattr(bot, "_cleanup"):
                try:
                    st.session_state.loop.run_until_complete(bot._cleanup())
                except:
                    pass
            raise

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║             BOT CREATION ERROR                   ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Nettoyage de la session
        if "bot_instance" in st.session_state:
            del st.session_state.bot_instance
        if "loop" in st.session_state:
            del st.session_state.loop

        return None


# Fonction d'aide pour la configuration asyncio
def setup_asyncio():
    """Configure l'environnement asyncio pour Streamlit"""
    try:
        if not st.session_state.get("loop"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            st.session_state.loop = loop
            nest_asyncio.apply()
        return st.session_state.loop
    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║             ASYNCIO SETUP ERROR                  ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
╚═════════════════════════════════════════════════╝
        """
        )
        return None


async def setup_streams(bot):
    """Configure and setup WebSocket streams"""
    try:
        tasks = []

        async def setup_single_stream(
            stream_type, setup_func, symbol="BTCUSDT", interval="1m"
        ):
            retry_count = 0
            while retry_count < WEBSOCKET_CONFIG["MAX_RETRIES"]:
                try:
                    logger.info(
                        f"Setting up {stream_type} stream (attempt {retry_count + 1}/{WEBSOCKET_CONFIG['MAX_RETRIES']})..."
                    )

                    # Création du socket avec le bon symbole
                    socket = None
                    if stream_type == "ticker":
                        socket = bot.socket_manager.trade_socket(symbol)
                    elif stream_type == "depth":
                        socket = bot.socket_manager.depth_socket(symbol)
                    elif stream_type == "kline":
                        socket = bot.socket_manager.kline_socket(symbol, interval)

                    if socket:
                        task = asyncio.create_task(
                            handle_socket_message(bot, socket, stream_type)
                        )
                        task.set_name(f"{stream_type}_stream_{symbol}")
                        return task

                    retry_count += 1
                    await asyncio.sleep(WEBSOCKET_CONFIG["RETRY_DELAY"])

                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error setting up {stream_type} stream: {e}")
                    if retry_count < WEBSOCKET_CONFIG["MAX_RETRIES"]:
                        await asyncio.sleep(WEBSOCKET_CONFIG["RETRY_DELAY"])
                    else:
                        logger.error(
                            f"Failed to setup {stream_type} stream after {WEBSOCKET_CONFIG['MAX_RETRIES']} attempts"
                        )
                        return None

        # Configuration des streams avec le bon ordre
        ticker_task = await setup_single_stream(
            "ticker", bot.socket_manager.trade_socket
        )
        depth_task = await setup_single_stream("depth", bot.socket_manager.depth_socket)
        kline_task = await setup_single_stream("kline", bot.socket_manager.kline_socket)

        # Collecte des tâches réussies
        tasks = [t for t in [ticker_task, depth_task, kline_task] if t is not None]

        if len(tasks) > 0:
            logger.info(
                f"✅ Successfully setup {len(tasks)}/{len(WEBSOCKET_CONFIG['STREAM_TYPES'])} streams"
            )
            return tasks
        else:
            logger.error("❌ Failed to setup any streams")
            return None

    except Exception as e:
        logger.error(f"❌ Stream setup error: {e}")
        return None


async def cleanup_existing_connections(bot):
    """Nettoie les connexions WebSocket existantes"""
    try:
        # Fermeture du socket manager
        if hasattr(bot, "socket_manager") and bot.socket_manager:
            try:
                # Fermeture des connexions WebSocket individuelles
                for socket_name in dir(bot.socket_manager):
                    if socket_name.startswith("_socket_"):
                        socket = getattr(bot.socket_manager, socket_name)
                        if hasattr(socket, "close"):
                            await socket.close()

                # Fermeture du socket manager lui-même
                if hasattr(bot.socket_manager, "close_connection"):
                    await bot.socket_manager.close_connection()

            except Exception as e:
                logger.warning(f"⚠️ Error closing socket manager: {e}")
            finally:
                bot.socket_manager = None

        # Fermeture du client WebSocket
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await bot.binance_ws.close_connection()
            except Exception as e:
                logger.warning(f"⚠️ Error closing Binance client: {e}")
            finally:
                bot.binance_ws = None

        return True

    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        return False


async def create_binance_client(bot):
    """
    Crée une nouvelle instance du client Binance

    Args:
        bot: Instance du bot de trading
    """
    try:
        # Création du client avec les credentials
        bot.binance_ws = await AsyncClient.create(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )

        # Création du socket manager
        bot.socket_manager = BinanceSocketManager(bot.binance_ws)

        return True

    except Exception as e:
        logger.error(f"❌ Error creating Binance client: {e}")
        return False


async def cleanup_resources(bot):
    """
    Nettoyage sécurisé des ressources avec protection de session et logging détaillé.

    Args:
        bot: Instance du bot de trading à nettoyer

    Returns:
        bool: True si le nettoyage a réussi, False sinon
    """
    current_time = datetime.now(timezone.utc)

    # Log de début de tentative de nettoyage
    logger.info(
        f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP ATTEMPT STARTED                ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Bot Status: {'Running' if st.session_state.get('bot_running') else 'Stopped'}
╚═════════════════════════════════════════════════╝
    """
    )

    # Vérification des conditions de protection
    protection_conditions = {
        "prevent_cleanup": st.session_state.get("prevent_cleanup", True),
        "keep_alive": st.session_state.get("keep_alive", True),
        "bot_running": st.session_state.get("bot_running", False),
        "ws_initializing": getattr(bot, "_ws_initializing", False),
        "bot_initialized": getattr(bot, "_initialized", False),
        "cleanup_in_progress": getattr(bot, "cleanup_in_progress", False),
        "force_cleanup": not st.session_state.get("force_cleanup", False),
        "cleanup_allowed": not st.session_state.get("cleanup_allowed", False),
    }

    # Si une condition de protection est active
    if any(protection_conditions.values()):
        # Log détaillé des conditions qui empêchent le nettoyage
        active_protections = [k for k, v in protection_conditions.items() if v]
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP PREVENTED                      ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Active Protections: {', '.join(active_protections)}
║ Session ID: {st.session_state.get('session_id', 'Unknown')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Renforcer la protection
        session_manager.protect_session()
        return False

    try:
        # Marquer le début du nettoyage
        bot.cleanup_in_progress = True
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP STARTED                        ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ WebSocket Status: {bot.ws_connection.get('status', 'unknown')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Fermeture du WebSocket
        await close_websocket(bot)

        # Log de succès
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP SUCCESSFUL                     ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Resources Cleaned: WebSocket, Buffer, Data
╚═════════════════════════════════════════════════╝
        """
        )
        return True

    except Exception as e:
        # Log d'erreur détaillé
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP ERROR                          ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        return False

    finally:
        # Nettoyage final et restauration de la protection
        try:
            bot.cleanup_in_progress = False
            session_manager.protect_session()

            # Log final
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP FINALIZED                      ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Protection Restored: True
║ Session Status: Protected
╚═════════════════════════════════════════════════╝
            """
            )

        except Exception as final_error:
            logger.error(f"Final cleanup error: {final_error}")


async def check_websocket_health(bot):
    """Vérifie l'état du WebSocket et le réinitialise si nécessaire"""
    try:
        # Vérifier si les streams sont actifs
        if not bot.ws_connection.get("tasks"):
            return await reset_websocket(bot)

        # Vérifier l'état des tâches
        active_tasks = [t for t in bot.ws_connection["tasks"] if not t.done()]
        if not active_tasks:
            return await reset_websocket(bot)

        # Vérifier si on reçoit des données
        if not bot.latest_data:
            return await reset_websocket(bot)

        return True

    except Exception as e:
        logger.error(f"❌ WebSocket health check error: {e}")
        await reset_websocket(bot)
        return False


async def close_websocket(bot):
    """Ferme proprement la connexion WebSocket"""
    try:
        logger.info("🔄 Closing WebSocket...")

        # Fermeture des tâches
        if bot.ws_connection and bot.ws_connection.get("tasks"):
            for task in bot.ws_connection["tasks"]:
                try:
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                except:
                    pass

        # Fermeture du socket manager
        if hasattr(bot, "socket_manager") and bot.socket_manager:
            try:
                await asyncio.wait_for(bot.socket_manager.close(), timeout=5.0)
            except:
                pass
            finally:
                bot.socket_manager = None

        # Fermeture du client websocket
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await asyncio.wait_for(bot.binance_ws.close_connection(), timeout=5.0)
            except:
                pass
            finally:
                bot.binance_ws = None

        # Fermeture explicite de la session client
        if hasattr(bot, "client_session") and bot.client_session:
            if not bot.client_session.closed:
                await bot.client_session.close()
                await asyncio.sleep(0.1)  # Petit délai pour assurer la fermeture
            bot.client_session = None

        # Réinitialisation de l'état
        bot.ws_connection = {"enabled": False, "status": "disconnected", "tasks": []}

        logger.info("✅ WebSocket closed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ WebSocket close error: {e}")
        return False


async def update_trading_data(bot):
    """Mise à jour des données de trading"""
    try:

        # Récupération des données BTC/USDC
        logger.info("📊 Récupération données pour BTC/USDC")
        btc_data = await fetch_market_data(bot, "BTCUSDT")
        if btc_data:
            bot.latest_data["BTCUSDT"] = btc_data

        # Récupération des données ETH/USDC
        logger.info("📊 Récupération données pour ETH/USDC")
        eth_data = await fetch_market_data(bot, "ETHUSDT")
        if eth_data:
            bot.latest_data["ETHUSDT"] = eth_data

    except Exception as e:
        logger.error(f"❌ Erreur mise à jour données: {e}")


async def handle_ticker_message(bot, msg):
    """Gestion des messages de ticker"""
    try:
        if "s" in msg and "p" in msg:
            symbol = msg["s"]
            price = float(msg["p"])

            # Mise à jour des données
            if not hasattr(bot, "latest_prices"):
                bot.latest_prices = {}
            bot.latest_prices[symbol] = price

            # Mise à jour du timestamp
            bot.ws_connection["last_message"] = time.time()

    except Exception as e:
        logger.error(f"❌ Ticker message error: {e}")


async def handle_kline_message(bot, msg):
    """Gestion des messages de klines"""
    try:
        if "k" in msg:
            kline = msg["k"]
            if all(k in kline for k in ["t", "o", "h", "l", "c", "v"]):
                candle = {
                    "timestamp": kline["t"],
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                }

                if not hasattr(bot, "latest_klines"):
                    bot.latest_klines = []
                bot.latest_klines.append(candle)

                if len(bot.latest_klines) > 1000:
                    bot.latest_klines.pop(0)

    except Exception as e:
        logger.error(f"❌ Kline message error: {e}")


async def handle_depth_message(bot, msg):
    """Gestion des messages d'orderbook"""
    try:
        if "a" in msg and "b" in msg:
            orderbook = {
                "asks": [[float(price), float(qty)] for price, qty in msg["a"]],
                "bids": [[float(price), float(qty)] for price, qty in msg["b"]],
                "timestamp": time.time(),
            }

            if not hasattr(bot, "latest_orderbook"):
                bot.latest_orderbook = {}
            bot.latest_orderbook = orderbook

    except Exception as e:
        logger.error(f"❌ Depth message error: {e}")


async def fetch_market_data(bot, symbol):
    """Récupère les données de marché de manière asynchrone"""
    try:
        # Configuration du timeframe par défaut si non défini
        if not hasattr(bot.config, "timeframe"):
            bot.config["timeframe"] = "1m"  # timeframe par défaut

        # Récupération des données via l'API Binance
        klines = await bot.binance_ws.get_klines(
            symbol=symbol, interval=bot.config["timeframe"]
        )

        # Conversion en format utilisable
        data = []
        for k in klines:
            candle = {
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            data.append(candle)

        return data

    except Exception as e:
        logger.error(f"❌ Erreur récupération données {symbol}: {e}")
        return None


async def update_market_data(bot):
    """Met à jour les données de marché"""
    try:
        data_received = False

        # Récupération BTC/USDC
        logger.info("📊 Récupération données pour BTC/USDC")
        btc_data = await fetch_market_data(bot, "BTCUSDT")
        if btc_data:
            bot.latest_data["BTCUSDT"] = btc_data
            data_received = True

        # Récupération ETH/USDC
        logger.info("📊 Récupération données pour ETH/USDC")
        eth_data = await fetch_market_data(bot, "ETHUSDT")
        if eth_data:
            bot.latest_data["ETHUSDT"] = eth_data
            data_received = True

        if not data_received:
            logger.warning("⚠️ Aucune donnée reçue")

        return data_received

    except Exception as e:
        logger.error(f"❌ Erreur mise à jour données: {e}")
        return False


async def process_market_data(bot, symbol):
    """Traite les données de marché pour un symbole"""
    try:
        data = bot.latest_data[symbol]
        if not data:
            return

        # Calcul des indicateurs
        if not hasattr(bot, "indicators"):
            bot.indicators = {}
        if symbol not in bot.indicators:
            bot.indicators[symbol] = {}

        # Mise à jour des indicateurs
        await update_indicators(bot, symbol, data)

        # Vérification des signaux
        await check_signals(bot, symbol)

    except Exception as e:
        logger.error(f"❌ Erreur traitement données {symbol}: {e}")


async def cleanup_session(bot):
    """Nettoyage d'une session avec verrou et cooldown"""
    global cleanup_in_progress, last_cleanup_time

    try:
        # Vérification du cooldown
        current_time = time.time()
        if current_time - last_cleanup_time < CLEANUP_COOLDOWN:
            return

        # Utilisation d'un verrou pour éviter les nettoyages simultanés
        async with cleanup_lock:
            if cleanup_in_progress:
                return

            cleanup_in_progress = True
            last_cleanup_time = current_time

            try:
                # Nettoyage des ressources
                await cleanup_resources(bot)

                # Un seul message de log
                logger.info("✅ Session cleaned successfully")
                logger.info(
                    """
╔═════════════════════════════════════════════════╗
║              CLEANUP COMPLETED                   ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned successfully              ║
╚═════════════════════════════════════════════════╝
                """
                )

            finally:
                cleanup_in_progress = False

    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


async def process_ws_message(bot, msg):
    """Process WebSocket messages"""
    try:
        if not msg:
            logger.warning("Empty message received")
            return

        if "e" not in msg:
            logger.warning(f"Invalid message format: {msg}")
            return

        if msg["e"] == "ticker":
            # Mise à jour du prix
            bot.latest_data["price"] = float(msg["c"])
            bot.latest_data["volume"] = float(msg["v"])
            logger.debug(f"💰 Price updated: {bot.latest_data['price']}")

        elif msg["e"] == "depth":
            # Mise à jour de l'orderbook
            bot.latest_data["orderbook"] = {"bids": msg["b"][:5], "asks": msg["a"][:5]}
            logger.debug("📚 Orderbook updated")

        elif msg["e"] == "kline":
            # Mise à jour des klines
            k = msg["k"]
            bot.latest_data["klines"] = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            logger.debug("📊 Klines updated")

        # Mise à jour du timestamp
        bot.latest_data["timestamp"] = msg.get("E", int(time.time() * 1000))
        bot.ws_connection["last_message"] = time.time()

    except Exception as e:
        logger.error(f"❌ Message processing error: {e}")


async def run_trading_bot():
    """Point d'entrée synchrone pour le bot de trading (statistiques uniquement, pas de bouton Start)"""
    try:
        # Stats en temps réel
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Portfolio Value", f"{portfolio_value:.2f} USDC", f"{pnl:+.2f} USDC"
            )
        with col2:
            st.metric("Active Positions", "2", "Open")
        with col3:
            st.metric("24h P&L", "+123 USDC", "+1.23%")

        # SUPPRESSION DU BOUTON Start Trading Bot !
        # Toute la logique de démarrage du bot doit être pilotée via la sidebar (main_async).

        # Tu peux afficher ici d'autres informations, ou l'état du bot, mais SANS bouton de démarrage.
        if st.session_state.get("bot_running"):
            st.success("🚀 Le trading bot est en cours d'exécution.")
        else:
            st.info("Le trading bot est arrêté. Utilise la sidebar pour le démarrer.")

    except Exception as e:
        logger.error(f"Trading bot error: {e}")
        st.error(f"❌ Trading bot error: {str(e)}")


async def main_async():
    """Point d'entrée principal de l'application avec gestion améliorée des états"""

    # --- DEBUG state au tout début ---
    def debug_state(when):
        import streamlit as st

        st.write(
            f"DEBUG [{when}] bot_running={st.session_state.get('bot_running')}, should_launch_bot={st.session_state.get('should_launch_bot')}, trading_task={st.session_state.get('trading_task')}"
        )
        print(
            f"DEBUG [{when}] bot_running={st.session_state.get('bot_running')}, should_launch_bot={st.session_state.get('should_launch_bot')}, trading_task={st.session_state.get('trading_task')}"
        )

    debug_state("DEBUT")

    # ---- 1. Initialisation des flags et objets critiques AVANT tout le reste ----
    if "bot" not in st.session_state or st.session_state["bot"] is None:
        st.session_state["bot"] = get_bot()
    bot = st.session_state["bot"]

    if "trading_task" not in st.session_state:
        st.session_state["trading_task"] = None
    if "should_launch_bot" not in st.session_state:
        st.session_state["should_launch_bot"] = False

    # ---- 2. Lancement du bot trading si flag actif (AVANT tout reset de state) ----
    if st.session_state.get("should_launch_bot", False):
        st.session_state["bot_running"] = True
        st.session_state["should_launch_bot"] = False  # reset le flag
        if not st.session_state.get("trading_task"):
            loop = st.session_state.get("loop") or asyncio.get_event_loop()
            st.session_state["trading_task"] = loop.create_task(
                bot.run_adaptive_trading(period="7d")
            )
    debug_state("APRES_LA_GESTION_LAUNCH_BOT")

    # ---- 3. Initialisation du reste du state par défaut ----
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    current_user = "Patmoorea"

    try:
        session_manager.protect_session()

        if "initialization_time" not in st.session_state:
            st.session_state.initialization_time = current_time

        default_session_state = {
            "session_id": f"{current_user}_{int(datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S').timestamp())}",
            "user": current_user,
            "portfolio": None,
            "latest_data": None,
            "indicators": None,
            # ATTENTION : on ne reset PAS bot_running ici pour ne pas casser le flag !
            "refresh_count": 0,
            "ws_status": "disconnected",
            "ws_initialized": False,
            "ws_connection_status": "disconnected",
            "last_update_time": current_time,
            "needs_update": False,
            "update_interval": 2.0,
            "last_refresh": time.time(),
        }
        for key, value in default_session_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

        if bot is None:
            st.error("❌ Failed to initialize bot")
            return

        await bot._initialize_analyzers()

        status_placeholder = st.sidebar.empty()

        # --- DEBUG données disponibles ---
        st.sidebar.markdown("#### Données présentes dans bot.latest_data :")

        latest_data = st.session_state.get("latest_data", {})
        if not isinstance(latest_data, dict):
            latest_data = {}
        st.sidebar.write(
            {k: getattr(v, "shape", str(type(v))) for k, v in latest_data.items()}
            if latest_data
            else "Aucune donnée"
        )

        # 5. Interface principale - État et contrôles
        status_col1, status_col2 = st.columns([2, 1])
        with status_col1:
            ws_status = st.session_state.get("ws_connection_status", "disconnected")
            ws_icon = {
                "connected": "🟢",
                "disconnected": "🔴",
                "initializing": "🔄",
                "error": "⚠️",
            }.get(ws_status, "🔴")

            status_info = f"""
            ### Bot Status
            - 🚦 Trading: {'🟢 Active' if st.session_state.get('bot_running') else '🔴 Stopped'}
            - 📡 WebSocket: {ws_icon} {ws_status.title()}
            - 💼 Portfolio: {'✅ Available' if st.session_state.portfolio else '⚠️ Not Available'}
            - ⏰ Last Update: {st.session_state.last_update_time}
            - 👤 User: {st.session_state.user}
            """
            st.info(status_info)

        # 6. Contrôles de la barre latérale avec gestion améliorée
        with st.sidebar:
            st.header("🛠️ Bot Controls")
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Low",
                key=f"risk_level_slider_{st.session_state.session_id}",
            )
            st.divider()

            # --- CONTROLES BOT TRADING ---
            if not st.session_state.get("bot_running", False):
                if st.button(
                    "🟢 Start Trading", key="start_button", use_container_width=True
                ):
                    st.session_state["should_launch_bot"] = True
                    st.rerun()  # ←←← AJOUT OBLIGATOIRE
                st.success("Cliquez pour démarrer le bot.")
            else:
                if st.button(
                    "🔴 Stop Trading", key="stop_button", use_container_width=True
                ):
                    st.session_state["bot_running"] = False
                    if st.session_state.get("trading_task"):
                        st.session_state["trading_task"].cancel()
                        st.session_state["trading_task"] = None
                    st.warning("Trading stoppé.")

            debug_state("APRES_BOUTONS")

            # --- AFFICHAGE LIVE ---
            if "live_status" in st.session_state and st.session_state["live_status"]:
                status_placeholder.markdown("### 🟢 Trading Live Status")
                for k, v in st.session_state["live_status"].items():
                    status_placeholder.write(f"**{k}** : {v}")

            # --- GESTION DES DONNEES ET BACKTEST ---
            latest_data = st.session_state.get("latest_data")
            if not isinstance(latest_data, dict):
                latest_data = {}
            st.write("DEBUG - latest_data:", latest_data)  # <-- À enlever ensuite

            def _has_valid_ohlcv(item):
                return (
                    isinstance(item, dict)
                    and "ohlcv" in item
                    and isinstance(item["ohlcv"], list)
                    and len(item["ohlcv"]) > 0
                    and isinstance(item["ohlcv"][0], dict)
                    and all(
                        k in item["ohlcv"][0]
                        for k in ["timestamp", "open", "high", "low", "close", "volume"]
                    )
                )

            data_ready = any(_has_valid_ohlcv(item) for item in latest_data.values())

            if not data_ready:
                st.warning(
                    "Aucune donnée OHLCV disponible. Clique sur le bouton ci-dessous pour charger les données de marché."
                )
                if st.button("Charger les données", key="load_data_btn"):
                    with st.spinner("Chargement des données..."):
                        loaded = False
                        try:
                            if not hasattr(bot, "binance_ws") or bot.binance_ws is None:
                                st.info("Initialisation de la WebSocket…")
                                await bot.initialize()
                            if hasattr(bot, "get_latest_data"):
                                data = await bot.get_latest_data()
                                st.write("DEBUG - Résultat get_latest_data:", data)
                                if data and isinstance(data, dict) and len(data) > 0:
                                    st.session_state["latest_data"] = data
                                    loaded = True
                                else:
                                    st.error(
                                        "La récupération a retourné None ou un dict vide : pas de données."
                                    )
                            elif hasattr(bot, "load_all_data"):
                                await bot.load_all_data()
                                latest_data = getattr(bot, "latest_data", {}) or {}
                                if not isinstance(latest_data, dict):
                                    latest_data = {}
                                st.write(
                                    "DEBUG - latest_data après load_all_data:",
                                    latest_data,
                                )
                                loaded = (
                                    isinstance(latest_data, dict)
                                    and len(latest_data) > 0
                                )
                                if loaded:
                                    st.session_state["latest_data"] = latest_data
                                else:
                                    st.error(
                                        "La récupération a retourné None ou un dict vide : pas de données."
                                    )
                            else:
                                st.error(
                                    "Aucune méthode de chargement trouvée sur le bot."
                                )
                        except Exception as exc:
                            st.error(f"Erreur lors du chargement des données : {exc}")
                        if loaded:
                            st.success("Données chargées ! Tu peux lancer un backtest.")
                            st.rerun()
            else:
                # --- BACKTEST CLASSIQUE ---
                if st.button("Lancer Backtest", key="backtest_all_btn"):
                    results = {}
                    st.info("Backtest en cours sur toutes les paires...")
                    try:
                        for symbol, data in latest_data.items():
                            try:
                                if _has_valid_ohlcv(data):
                                    import pandas as pd

                                    df = pd.DataFrame(data["ohlcv"])

                                    def strategy_func(df, **params):
                                        return (
                                            df["close"] > df["close"].rolling(5).mean()
                                        ).astype(int)

                                    engine = BacktestEngine(initial_capital=10000)
                                    results[symbol] = engine.run_backtest(
                                        df, strategy_func
                                    )
                                else:
                                    st.warning(
                                        f"Aucune donnée OHLCV exploitable pour {symbol}"
                                    )
                            except Exception as pair_exc:
                                st.warning(f"Erreur sur {symbol}: {pair_exc}")
                        st.session_state["all_backtest_results"] = results
                        st.success("Backtest terminé ✅")
                    except Exception as batch_exc:
                        st.error(f"Erreur lors du backtest: {batch_exc}")

                # Résultats
                if st.session_state.get("all_backtest_results"):
                    st.markdown("**Résultats Backtest Classique :**")
                    for symbol, res in st.session_state["all_backtest_results"].items():
                        st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")

                    if st.button(
                        "Lancer Backtest Quantique", key="quantum_backtest_all_btn"
                    ):
                        st.info("Backtest quantique en cours sur toutes les paires...")
                        results = {}
                        if not isinstance(latest_data, dict):
                            latest_data = {}
                        st.write(
                            "DEBUG - Paire/Data dispo :",
                            {
                                k: getattr(v, "shape", str(type(v)))
                                for k, v in latest_data.items()
                            },
                        )
                        try:
                            for symbol, data in latest_data.items():
                                st.write(f"Test {symbol} ...")
                                try:
                                    if _has_valid_ohlcv(data):
                                        import pandas as pd

                                        df = pd.DataFrame(data["ohlcv"])

                                        def strategy_func(df, **params):
                                            return (
                                                df["close"]
                                                > df["close"].rolling(5).mean()
                                            ).astype(int)

                                        engine = BacktestEngine(initial_capital=10000)
                                        results[symbol] = engine.run_backtest(
                                            df, strategy_func
                                        )
                                    else:
                                        st.warning(
                                            f"Aucune donnée OHLCV exploitable pour {symbol}"
                                        )
                                except Exception as pair_exc:
                                    st.warning(
                                        f"Erreur quantique sur {symbol}: {pair_exc}"
                                    )
                            st.session_state["all_quantum_results"] = results
                            st.success("Backtest quantique terminé ✅")
                            st.write("DEBUG - Résultats quantum :", results)
                        except Exception as batch_exc:
                            st.error(f"Erreur lors du backtest quantique: {batch_exc}")

                if st.session_state.get("all_quantum_results"):
                    st.markdown("**Résultats Backtest Quantique :**")
                    for symbol, res in st.session_state["all_quantum_results"].items():
                        st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")

        # 8. Onglets principaux avec gestion d'erreur
        try:
            portfolio_tab, trading_tab, analysis_tab = st.tabs(
                ["📈 Portfolio", "🎯 Trading", "📊 Analysis"]
            )

            # Onglet Portfolio
            with portfolio_tab:
                await _render_portfolio_tab(bot)

            # Onglet Trading
            with trading_tab:
                await _render_trading_tab(bot)

            # Onglet Analysis
            with analysis_tab:
                await _render_analysis_tab(bot)
        except Exception as tab_error:
            logger.error(f"Tab rendering error: {tab_error}")
            st.error("Error rendering tabs")

    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        st.error(f"❌ Application error: {str(e)}")
    finally:
        # Protection finale avec timestamp
        try:
            session_manager.protect_session()
            st.session_state.last_update_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception as protect_error:
            logger.error(f"Session protection error: {protect_error}")

    debug_state("FIN")


# Fonctions auxiliaires pour le rendu des onglets
async def _render_portfolio_tab(bot):
    """Rendu de l'onglet Portfolio"""
    if st.session_state.bot_running:
        try:
            portfolio = st.session_state.get("portfolio")
            if portfolio:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "💰 Total Value",
                        f"{portfolio.get('total_value', 0):.2f} USDC",
                        f"{portfolio.get('daily_pnl', 0):+.2f} USDC",
                    )
                with col2:
                    st.metric(
                        "📈 24h Volume",
                        f"{portfolio.get('volume_24h', 0):.2f} USDC",
                        f"{portfolio.get('volume_change', 0):+.2f}%",
                    )
                with col3:
                    positions = portfolio.get("positions", [])
                    st.metric(
                        "🔄 Active Positions",
                        str(len(positions)),
                        f"{len(positions)} active",
                    )

                if positions:
                    st.subheader("Active Positions")
                    st.dataframe(pd.DataFrame(positions), use_container_width=True)
                else:
                    st.info("💡 No active positions")
            else:
                st.warning("⚠️ Waiting for portfolio data...")
        except Exception as e:
            st.error(f"❌ Portfolio error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view portfolio")


async def _render_trading_tab(bot):
    """Rendu de l'onglet Trading"""
    if st.session_state.bot_running:
        try:
            latest_data = bot.latest_data.get("BTCUSDT", {})
            if latest_data:
                col1, col2 = st.columns(2)
                with col1:
                    current_price = latest_data[-1]["close"]
                    prev_price = (
                        latest_data[-2]["close"]
                        if len(latest_data) > 1
                        else current_price
                    )
                    price_change = (
                        ((current_price - prev_price) / prev_price * 100)
                        if prev_price
                        else 0
                    )

                    st.metric(
                        "BTC/USDC Price",
                        f"{current_price:.2f}",
                        f"{price_change:+.2f}%",
                    )
                with col2:
                    current_vol = latest_data[-1]["volume"]
                    prev_vol = (
                        latest_data[-2]["volume"]
                        if len(latest_data) > 1
                        else current_vol
                    )
                    vol_change = (
                        ((current_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
                    )

                    st.metric(
                        "Trading Volume", f"{current_vol:.2f}", f"{vol_change:+.2f}%"
                    )

            if bot.indicators:
                st.subheader("Trading Signals")
                st.dataframe(pd.DataFrame(bot.indicators), use_container_width=True)
            else:
                st.info("💡 Waiting for signals...")
        except Exception as e:
            st.error(f"❌ Trading data error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view signals")


async def _render_analysis_tab(bot):
    """Rendu de l'onglet Analysis"""
    if st.session_state.bot_running:
        try:
            if bot.latest_data and bot.indicators:
                st.subheader("Technical Analysis")

                for symbol in bot.latest_data:
                    await process_market_data(bot, symbol)

                if hasattr(bot, "advanced_indicators"):
                    analysis = bot.advanced_indicators.get_all_signals()
                    st.dataframe(pd.DataFrame(analysis), use_container_width=True)
                else:
                    st.info("💡 Processing analysis...")
            else:
                st.info("💡 Waiting for market data...")
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view analysis")

    # --- Signal Quantum SVM ---
    if hasattr(bot, "qsvm") and bot.qsvm is not None:
        try:
            # Prépare les features à passer à predict (adapte cette ligne selon ta logique)
            features = (
                bot.latest_data
            )  # ou bot.indicators ou ton dataframe, adapte selon besoin
            quantum_signal = bot.qsvm.predict(features)
            st.subheader("Quantum SVM Signal")
            st.metric("Quantum SVM Signal", quantum_signal)
        except Exception as e:
            st.warning(f"Erreur Quantum SVM : {e}")


async def shutdown():
    """Arrêt propre de l'application"""
    try:
        # Récupération des tâches en cours
        tasks = [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]

        if tasks:
            # Annulation des tâches
            for task in tasks:
                task.cancel()

            # Attente de la fin des tâches avec timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout during tasks cancellation")

        # Nettoyage via le gestionnaire de sessions
        await session_manager.cleanup()

        # Nettoyage des ressources du bot
        if "bot_instance" in st.session_state:
            bot = st.session_state.bot_instance
            await cleanup_resources(bot)

        logger.info(
            """
╔═════════════════════════════════════════════════╗
║              SHUTDOWN COMPLETED                  ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned and sessions closed       ║
╚═════════════════════════════════════════════════╝
        """
        )

    except Exception as e:
        logger.error(f"Shutdown error: {e}")


def main():
    """Point d'entrée principal avec protection renforcée et gestion des événements améliorée"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")

    try:
        # 1. Initialisation et protection de la session
        global session_manager
        session_manager = StreamlitSessionManager()
        session_manager.protect_session()

        # 2. Log de démarrage
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              STARTING APPLICATION                ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
║ Session: {session_manager.session_id}
╚═════════════════════════════════════════════════╝
        """
        )

        # 3. Initialisation de l'état de session
        _initialize_session_state()

        # 4. Configuration et vérification de la boucle d'événements
        event_loop = _setup_and_verify_event_loop()
        if not event_loop:
            raise RuntimeError("Failed to initialize event loop")

        # 5. Exécution de la coroutine principale
        event_loop.run_until_complete(main_async())

    except asyncio.CancelledError:
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              GRACEFUL SHUTDOWN                   ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
╚═════════════════════════════════════════════════╝
        """
        )

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              RUNTIME ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
╚═════════════════════════════════════════════════╝
        """
        )
        st.error(f"❌ Application error: {str(e)}")

    finally:
        _perform_cleanup()


def _initialize_session_state():
    """Initialise l'état de la session avec des valeurs sûres et logging détaillé"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")
    session_id = f"{current_user}_{int(current_time.timestamp())}"

    try:
        # États par défaut avec horodatage
        default_state = {
            # États de base
            "session_id": session_id,
            "initialization_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "user": current_user,
            "initialized": True,
            # États du bot
            "bot_running": False,
            "portfolio": None,
            "latest_data": {},
            "indicators": None,
            "refresh_count": 0,
            # États de la boucle événementielle
            "loop": None,
            "error_count": 0,
            # États WebSocket
            "ws_status": "disconnected",
            "ws_initialized": False,
            "ws_connection_status": "disconnected",
            "ws_last_heartbeat": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            # Protections
            "keep_alive": True,
            "prevent_cleanup": True,
            "force_cleanup": False,
            "cleanup_allowed": False,
        }

        # Initialisation des états manquants uniquement
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Log de succès
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION STATE INITIALIZED              ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
║ Session ID: {session_id}
║ Status: Active
╚═════════════════════════════════════════════════╝
        """
        )

        return True

    except Exception as e:
        # Log d'erreur
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║           SESSION STATE ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
╚═════════════════════════════════════════════════╝
        """
        )
        return False


def _setup_and_verify_event_loop():
    """Configure et vérifie la boucle d'événements avec gestion d'erreur améliorée"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")

    try:
        # Vérification de l'existence d'une boucle
        if not st.session_state.get("loop"):
            # Création et configuration de la nouvelle boucle
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply()

            # Sauvegarde dans la session
            st.session_state.loop = loop

            # Log de succès d'initialisation
            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP INITIALIZED              ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
║ Status: Successfully configured
║ Loop ID: {id(loop)}
╚═════════════════════════════════════════════════╝
            """
            )

            return loop

        # Vérification de la boucle existante
        existing_loop = st.session_state.loop
        if existing_loop.is_closed():
            logger.warning(
                f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP CLOSED                   ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Status: Creating new loop
║ Previous Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
            """
            )

            # Création d'une nouvelle boucle
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            nest_asyncio.apply()
            st.session_state.loop = new_loop
            return new_loop

        # Retour de la boucle existante
        logger.debug(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP VERIFIED                 ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Status: Using existing loop
║ Loop ID: {id(existing_loop)}
╚═════════════════════════════════════════════════╝
        """
        )

        return existing_loop

    except Exception as e:
        # Log d'erreur détaillé
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
║ Details: {traceback.format_exc()}
╚═════════════════════════════════════════════════╝
        """
        )

        # Incrément du compteur d'erreurs
        st.session_state.error_count = st.session_state.get("error_count", 0) + 1

        return None

    finally:
        # Mise à jour du timestamp
        st.session_state.last_update_time = current_time.strftime("%Y-%m-%d %H:%M:%S")


def _perform_cleanup():
    """Effectue le nettoyage final de l'application"""
    try:
        # 1. Protection de la session
        session_manager.protect_session()

        # 2. Nettoyage de la boucle d'événements
        if st.session_state.get("loop"):
            loop = st.session_state.loop
            if not loop.is_closed():
                try:
                    # Nettoyage conditionnel des ressources
                    if st.session_state.get(
                        "force_cleanup", False
                    ) and st.session_state.get("cleanup_allowed", False):
                        if "bot_instance" in st.session_state:
                            loop.run_until_complete(
                                cleanup_resources(st.session_state.bot_instance)
                            )
                    # NE PAS FERMER LA BOUCLE ! On ne fait PAS loop.close()
                except Exception as e:
                    logger.error(f"Loop cleanup error: {e}")
                finally:
                    # On ne détruit pas la boucle ici non plus
                    pass

        logger.info(
            """
╔═════════════════════════════════════════════════╗
║              CLEANUP COMPLETED                   ║
╠═════════════════════════════════════════════════╣
║ Status: All resources cleaned
╚═════════════════════════════════════════════════╝
        """
        )

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              CLEANUP ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
    finally:
        # Protection finale absolue
        session_manager.protect_session()


def ensure_event_loop():
    """Vérifie et assure l'existence d'une boucle d'événements valide"""
    try:
        if not st.session_state.get("loop"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply()
            st.session_state.loop = loop

            logger.info("✅ New event loop created and configured")
            return loop

        return st.session_state.loop

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              EVENT LOOP ERROR                    ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        return None


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              KEYBOARD INTERRUPT                  ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Status: Graceful shutdown initiated
╚═════════════════════════════════════════════════╝
        """
        )

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              CRITICAL ERROR                      ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        sys.exit(1)

    finally:
        try:
            # Nettoyage final avec nouvelle boucle si nécessaire
            if "bot_instance" in st.session_state:
                try:
                    # Création d'une nouvelle boucle pour le nettoyage final
                    cleanup_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(cleanup_loop)
                    cleanup_loop.run_until_complete(
                        cleanup_resources(st.session_state.bot_instance)
                    )
                    cleanup_loop.close()
                except Exception as e:
                    logger.error(f"Final cleanup error: {e}")

            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║              FINAL CLEANUP                       ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Status: All resources cleaned
╚═════════════════════════════════════════════════╝
            """
            )

        except Exception as cleanup_error:
            logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║              CLEANUP ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Error: {str(cleanup_error)}
╚═════════════════════════════════════════════════╝
            """
            )
