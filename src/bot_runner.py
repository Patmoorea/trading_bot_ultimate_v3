import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import warnings
import logging
import json
import asyncio
import aiohttp
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import argparse
import numpy as np
import pandas as pd
import pandas_ta as ta
from decimal import Decimal
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.analysis.news.cointelegraph_fetcher import fetch_cointelegraph_news

# Obtenir le chemin racine du projet (un niveau au-dessus de l'emplacement du script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import des modules existants avec les bons chemins
from web_interface.app.services.order_execution import SmartOrderExecutor
from src.strategies.arbitrage.execution.execution import ArbitrageExecutor
from src.ai.enhanced_cnn_lstm import EnhancedCNNLSTM
from src.ai_models.hybrid.cnn_lstm_enhanced import EnhancedCNNLSTM
from src.ai.ppo_gtrxl import PPOGTrXL
from src.analysis.news.sentiment_analyzer import NewsSentimentAnalyzer
from src.connectors.binance import BinanceConnector
from src.ai.deep_learning_model import DeepLearningModel
from src.ai.ppo_strategy import PPOStrategy
from src.bot.trading_env import TradingEnv

from src.strategies.arbitrage.core.arbitrage_bot import ArbitrageBot
from src.strategies.arbitrage.multi_exchange.arbitrage_scanner import ArbitrageScanner
from src.strategies.arbitrage.core.risk_management.risk_manager import RiskManager
from src.strategies.arbitrage.service import ArbitrageEngine

from src.data.ws_buffered_collector import BufferedWSCollector

from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators

from src.optimization.optuna_wrapper import (
    tune_hyperparameters,
    optimize_hyperparameters_full,
)
from src.security.key_manager import KeyManager

from src.backtesting.core.backtest_engine import BacktestEngine

# Import dynamique des stratégies
from src.strategies import sma_strategy, breakout_strategy, arbitrage_strategy

# Charger les variables d'environnement depuis .env
load_dotenv()

def fetch_binance_ohlcv(
    symbol, interval, start_str, end_str=None, api_key=None, api_secret=None
):
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    if not klines or len(klines) == 0:
        print(f"Aucune donnée récupérée pour {symbol}")
        return None
    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtest", action="store_true", help="Lancer un backtest quantitatif"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/historical/BTCUSDT_1h.csv",
        help="Chemin du CSV market data",
    )
    parser.add_argument("--capital", type=float, default=10000, help="Capital initial")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sma",
        choices=["sma", "breakout", "arbitrage"],
        help="Stratégie à utiliser",
    )
    # Ajoute ici d'autres paramètres si besoin (fast_window, slow_window, lookback, etc.)

    args = parser.parse_args()

    if args.backtest:
        print("=== Lancement du backtesting quantitatif ===")
        df = pd.read_csv(args.data)

        # Choix de la stratégie
        strategy_map = {
            "sma": sma_strategy,
            "breakout": breakout_strategy,
            "arbitrage": arbitrage_strategy,
        }
        strategy_func = strategy_map[args.strategy]

        # Exemple : utilise des paramètres par défaut, ou récupère-les via argparse
        results = BacktestEngine(initial_capital=args.capital).run_backtest(
            df, strategy_func
        )
        print("Résultats backtest :")
        print(results)
        exit(0)


def debug_market_data_structure(market_data, pairs_valid, timeframes):
    for pair in pairs_valid:
        pair_key = pair.replace("/", "").upper()
        if pair_key not in market_data:
            # print(f"  ❌ ABSENT de market_data")
            continue
        for tf in timeframes:
            tf_data = market_data[pair_key].get(tf)
            if tf_data is None:
                # print(f"  - {tf}: ❌ ABSENT")
                pass
            elif isinstance(tf_data, dict):
                # print(f"  - {tf}: OK, keys: {list(tf_data.keys())}")
                pass
            else:
                # print(f"  - {tf}: Type inattendu: {type(tf_data)}")
                pass


# Charger les tokens Telegram depuis .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ Attention: Variables Telegram non trouvées dans .env")

# Configuration ULTRA-stricte pour éliminer TOUS les warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["STREAMLIT_HIDE_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Supprimer TOUS les warnings Python
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Configuration logging pour ne montrer que nos messages
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Constantes de marché
MARKET_REGIMES = {
    "TRENDING_UP": "Tendance Haussière",
    "TRENDING_DOWN": "Tendance Baissière",
    "RANGING": "Range/Scalping",
    "VOLATILE": "Haute Volatilité",
}


def get_current_time():
    """Retourne le temps actuel au format UTC"""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# Constantes
CURRENT_TIME = get_current_time()
CURRENT_USER = "Patmoorea"
CONFIG_PATH = "config/trading_pairs.json"
SHARED_DATA_PATH = "src/shared_data.json"


def safe(val, default="N/A", fmt=None):
    """Sécurise l'affichage d'une valeur (None => défaut, format optionnel)"""
    try:
        if val is None:
            return default
        if fmt:
            return fmt.format(val)
        return val
    except Exception:
        return default


class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        if not bot_token or not chat_id:
            print("⚠️ Configuration Telegram incomplète")
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send_message(self, message):
        """Envoie un message sur Telegram"""
        if not self.bot_token or not self.chat_id:
            print("⚠️ Message non envoyé: Configuration Telegram manquante")
            return

        header = (
            f"🕒 {get_current_time()}\n"
            f"👤 {CURRENT_USER}\n"
            "------------------------\n"
        )
        full_message = header + message

        url = f"{self.base_url}/sendMessage"
        data = {"chat_id": self.chat_id, "text": full_message, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        print(f"⚠️ Erreur Telegram: {result.get('description')}")
                    return result
        except Exception as e:
            print(f"⚠️ Erreur envoi Telegram: {e}")

    async def send_performance_update(self, performance_data):
        message = (
            "🤖 <b>Trading Bot Status Update</b>\n\n"
            f"💰 Balance: ${safe(performance_data.get('balance'))}\n"
            f"📊 Win Rate: {safe(performance_data.get('win_rate', 0)*100, 'N/A', '{:.1f}')}%\n"
            f"📈 Profit Factor: {safe(performance_data.get('profit_factor'))}\n"
            f"🔄 Total Trades: {safe(performance_data.get('total_trades'), 'N/A', '{:d}')}\n"
        )
        await self.send_message(message)

    async def send_cycle_update(self, cycle, regime, duration):
        """Envoie une mise à jour du cycle"""
        message = (
            "🔄 <b>Cycle Update</b>\n\n"
            f"📊 Cycle: {cycle}\n"
            f"🎯 Régime: {regime}\n"
            f"⏱️ Durée: {duration:.1f}s\n"
        )
        await self.send_message(message)

    async def send_trade_alert(self, trade_data):
        """Envoie une alerte de trade sur Telegram"""
        emoji = "🟢" if trade_data["side"] == "BUY" else "🔴"
        message = (
            f"{emoji} <b>Trade {trade_data['side']}</b>\n\n"
            f"📊 Paire: {trade_data['symbol']}\n"
            f"💰 Quantité: {trade_data['amount']}\n"
            f"💵 Prix: {trade_data.get('price', 'Market')}\n"
            f"🕒 Heure: {get_current_time()}\n"
            f"📝 Raison: {trade_data.get('reason', 'Signal de trading')}"
        )
        if not self.ai_enabled:
            print("⚠️ IA désactivée. Raison possible:")
            print("- Modèle Deep Learning:", "OK" if self.dl_model else "❌")
            print("- Stratégie PPO:", "OK" if self.ppo_strategy else "❌")
            print("- Environnement:", "OK" if self.env else "❌")
        await self.send_message(message)

    async def send_arbitrage_alert(self, opportunity):
        """Envoie une alerte d'opportunité d'arbitrage"""
        message = (
            f"🔄 <b>Opportunité d'Arbitrage</b>\n\n"
            f"📊 Paire: {opportunity['pair']}\n"
            f"💹 Différence: {opportunity['diff_percent']:.2f}%\n"
            f"📈 {opportunity['exchange1']}: {opportunity['price1']}\n"
            f"📉 {opportunity['exchange2']}: {opportunity['price2']}\n"
            f"💰 Profit potentiel: {(opportunity['diff_percent'] - 0.2):.2f}% (après frais)"
        )
        await self.send_message(message)

    async def send_news_summary(
        self,
        news_data,
        market_data=None,
        ai_summary: str = None,
        filter_symbols=None,
        filter_volatility=None,
    ):
        if not news_data or len(news_data) == 0:
            await self.send_message(
                "📰 <b>Dernières Nouvelles Importantes</b>\n\nAucune news significative détectée récemment."
            )
            return

        # Mapping emoji par source
        source_emoji = {
            "CoinDesk": "📰",
            "Cointelegraph": "🟣",
            "Decrypt": "🟦",
            "Binance": "🟡",
            "Twitter": "🐦",
            "default": "🗞️",
        }

        # Filtrage avancé selon symboles ou volatilité
        filtered_news = []
        for news in news_data:
            # Filtrage par symbole
            if filter_symbols:
                news_symbols = [s.upper() for s in news.get("symbols", [])]
                if not any(sym in news_symbols for sym in filter_symbols):
                    continue
            # Filtrage par volatilité
            if filter_volatility and market_data and news.get("symbols"):
                symbol = news["symbols"][0].replace("/", "")
                vol = market_data.get(symbol, {}).get("1h", {}).get("volatility", 0)
                if vol is not None and vol < filter_volatility:
                    continue
            filtered_news.append(news)

        # Si rien ne passe le filtre, utilise tout
        if not filtered_news:
            filtered_news = news_data

        message = "📰 <b>Dernières Nouvelles Importantes</b>\n\n"

        # Ajoute le résumé IA si fourni
        if ai_summary:
            message += f"🤖 <b>Résumé IA:</b>\n{ai_summary}\n\n"

        for news in filtered_news[:5]:
            # Source
            src = news.get("source", "default")
            emoji = source_emoji.get(src, source_emoji["default"])
            # Lien cliquable sur le titre
            title = news.get("title", "NO_TITLE")
            url = news.get("url", "")
            if url:
                title_line = f'{emoji} <a href="{url}">{title}</a>'
            else:
                title_line = f"{emoji} {title}"

            # Source affichée en fin de ligne
            title_line += f" <i>({src})</i>\n"

            # Résumé court
            summary_text = news.get("summary") or news.get("text") or ""
            if summary_text:
                message += f"{title_line}└ {summary_text[:180]}...\n\n"
            else:
                message += f"{title_line}\n"

        await self.send_message(message)


class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, message):
        if any(
            word in message()
            for word in [
                "warning",
                "scriptruncontext",
                "missing",
                "streamlit",
                "pair",
                "not available",
                "skipping",
            ]
        ):
            return
        self.original_stderr.write(message)

    def flush(self):
        self.original_stderr.flush()


sys.stderr = WarningFilter(sys.stderr)


class TradingBotM4:
    def __init__(self):
        # Configuration de base existante...
        self.config = {
            "TRADING": {
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "pairs": ["BTC/USDT", "ETH/USDT"],
            },
            "AI": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "n_epochs": 10,
                "verbose": 1,
            },
        }
        # >>>> AJOUT ICI <<<<
        self.ws_collector = BufferedWSCollector(
            symbols=[
                s.replace("/", "").upper() for s in self.config["TRADING"]["pairs"]
            ],
            timeframes=self.config["TRADING"]["timeframes"],
            maxlen=2000,
        )
        # Initialize basic attributes...
        self.data_file = SHARED_DATA_PATH
        self.current_cycle = 0
        self.regime = MARKET_REGIMES["RANGING"]
        self.market_data = {}
        self.indicators = {}
        self.ai_weight = 0.3
        self.ai_enabled = False

        # Initialisation de l'arbitrage engine
        try:
            self.arbitrage_engine = ArbitrageEngine()
            self.brokers = self.arbitrage_engine.brokers
            print("✅ ArbitrageEngine initialisé avec succès")
        except Exception as e:
            print(f"⚠️ Erreur initialisation ArbitrageEngine: {e}")
            self.arbitrage_engine = None
            self.brokers = {}

        self.arbitrage_executor = ArbitrageExecutor(self.brokers)

        # Initialisation de l'environnement (une seule fois)
        print("Configuration de l'environnement...")
        self.env = TradingEnv(
            trading_pairs=self.config["TRADING"]["pairs"],
            timeframes=self.config["TRADING"]["timeframes"],
        )
        print("✅ Environnement initialisé avec succès")

        # Initialisation de l'IA
        self._initialize_ai()

        # Initialize shared data
        self.initialize_shared_data()

        # Initialize basic attributes
        self.data_file = SHARED_DATA_PATH
        self.current_cycle = 0
        self.regime = MARKET_REGIMES["RANGING"]
        self.market_data = {}
        self.indicators = {}
        self.ai_weight = 0.3  # Add AI weight initialization
        self.ai_enabled = False  # Initialize AI status
        self.pairs_valid = []
        # Initialize shared data
        self.initialize_shared_data()

        # Initialize environment
        print("Checking environment setup...")
        self.env = TradingEnv(
            trading_pairs=self.config["TRADING"]["pairs"],
            timeframes=self.config["TRADING"]["timeframes"],
        )
        print(f"Trading pairs: {self.pairs_valid}")
        print(f"Environment initialized: {hasattr(self, 'env')}")
        if hasattr(self, "env"):
            print(
                f"Environment methods: reset={hasattr(self.env, 'reset')}, step={hasattr(self.env, 'step')}"
            )
        # Initialisation des composants d'arbitrage
        self.arbitrage_bot = ArbitrageBot()
        self.arbitrage_scanner = ArbitrageScanner()
        self.risk_manager = RiskManager()

        # Configuration de l'arbitrage
        self.arbitrage_config = {
            "min_profit": 0.5,  # 0.5% minimum profit
            "max_exposure": 10000,  # Maximum exposure in USDC
            "enabled_exchanges": ["binance", "kucoin", "huobi"],
        }
        # Sécurité avancée: gestion de clé cold wallet
        self.key_manager = KeyManager()
        if not self.key_manager.has_key():
            print(
                "Aucune clé cold wallet détectée, génération d'une nouvelle clé sécurisée…"
            )
            pk = self.key_manager.generate_private_key()
            self.key_manager.save_private_key()
            print("Clé cold wallet générée et sauvegardée de manière chiffrée.")
        else:
            try:
                self.key_manager.load_private_key()
                print("Clé cold wallet chargée avec succès.")
            except Exception as e:
                print(f"Erreur de chargement de la clé cold wallet: {e}")

    def get_binance_real_balance(self, asset="USDC"):
        if self.is_live_trading and self.binance_client:
            try:
                balance_info = self.binance_client.get_asset_balance(asset=asset)
                if balance_info:
                    return float(balance_info["free"])
            except Exception as e:
                self.logger.error(f"Erreur récupération balance Binance: {e}")
        return None

    async def detect_arbitrage_opportunities(self, pair=None):
        """Détecte les opportunités d'arbitrage avec vérification des volumes"""
        if not self.is_live_trading:
            return []

        opportunities = []
        pairs_to_check = [pair] if pair else self.pairs_valid
        MIN_PROFIT_THRESHOLD = 0.15
        MIN_VOLUME_USD = 10000
        MAX_SPREAD = 0.5

        try:
            for current_pair in pairs_to_check:
                try:
                    pair_key = current_pair.replace("/", "")

                    # Liste des échanges réels à comparer
                    exchanges_to_check = [
                        {"name": "okx", "client": self.brokers.get("okx")},
                        {"name": "gateio", "client": self.brokers.get("gateio")},
                        {"name": "blofin", "client": self.brokers.get("blofin")},
                        {"name": "bingx", "client": self.brokers.get("bingx")},
                    ]

                    # Prix Binance comme référence (doit être await si async)
                    binance_ticker = (
                        await self.binance_client.fetch_ticker(pair_key)
                        if hasattr(self.binance_client, "fetch_ticker")
                        and asyncio.iscoroutinefunction(
                            self.binance_client.fetch_ticker
                        )
                        else self.binance_client.get_ticker(symbol=pair_key)
                    )
                    binance_price = float(
                        binance_ticker.get("lastPrice")
                        or binance_ticker.get("last")
                        or 0
                    )
                    binance_volume = float(binance_ticker.get("volume", 0))

                    # Comparaison avec les autres échanges
                    for exchange in exchanges_to_check:
                        if not exchange["client"]:
                            continue

                        try:
                            # Récupération du prix sur l'autre échange (toujours await!)
                            ticker = await exchange["client"].fetch_ticker(current_pair)
                            exchange_price = ticker["last"]
                            if not exchange_price or not binance_price:
                                continue

                            # Calcul de la différence de prix
                            price_diff = abs(exchange_price - binance_price)
                            profit_pct = (
                                (price_diff / binance_price) * 100
                                if binance_price > 0
                                else 0
                            )

                            if profit_pct > MIN_PROFIT_THRESHOLD:
                                opportunity = {
                                    "pair": current_pair,
                                    "exchange1": "Binance",
                                    "exchange2": exchange["name"],
                                    "price1": binance_price,
                                    "price2": exchange_price,
                                    "diff_percent": profit_pct,
                                    "volume_24h": binance_volume * binance_price,
                                    "estimated_profit": profit_pct - 0.2,  # Après frais
                                }
                                opportunities.append(opportunity)
                                self.logger.info(
                                    f"Opportunité d'arbitrage détectée pour {current_pair}: {opportunity}"
                                )

                        except Exception as e:
                            self.logger.error(f"Erreur sur {exchange['name']}: {e}")
                            continue

                except Exception as e:
                    self.logger.error(
                        f"Erreur lors du traitement de {current_pair}: {e}"
                    )
                    continue

            return opportunities

        except Exception as e:
            self.logger.error(f"Erreur globale détection arbitrage: {e}")
            return []

    async def execute_arbitrage(self, opportunity):
        """Exécute une opportunité d'arbitrage"""
        try:
            # Utiliser l'ArbitrageExecutor existant
            result = await self.arbitrage_executor.execute(
                opportunity=opportunity,
                max_slippage=0.1,  # 0.1% de slippage maximum
                timeout=5,  # 5 secondes maximum
            )

            if result["success"]:
                profit = result["realized_profit"]
                message = (
                    f"✅ Arbitrage réussi!\n"
                    f"💰 Profit: {profit:.2f} USDT\n"
                    f"📊 Paire: {opportunity['pair']}\n"
                    f"🔄 Route: {opportunity['route']}"
                )
                await self.telegram.send_message(message)

                # Mettre à jour les statistiques
                self._update_performance_metrics(
                    {"type": "arbitrage", "profit": profit, "pair": opportunity["pair"]}
                )
            else:
                self.logger.warning(f"Échec arbitrage: {result['error']}")

        except Exception as e:
            self.logger.error(f"Erreur exécution arbitrage: {e}")

        def secure_withdraw(self, address, amount, asset):
            # Cette fonction serait appelée avant tout transfert sortant
            # Demande la signature de l'opération
            message = f"{address}|{amount}|{asset}|{get_current_time()}"
            signature = self.key_manager.sign_message(message)
            # Ici, tu pourrais envoyer la requête à l'exchange avec la signature pour logs/sécurité
            print(
                f"Retrait sécurisé demandé : {amount} {asset} vers {address}, signature: {signature}"
            )
            # (A compléter: appel API exchange avec signature, ou log d'audit)
            return signature

    def _initialize_ai(self):
        """Initialise les composants d'IA"""
        try:
            print("Initialisation des modèles d'IA...")

            if not self.env:
                raise ValueError("L'environnement de trading n'est pas initialisé")

            # Initialisation du modèle Deep Learning
            self.dl_model = DeepLearningModel()

            # Configuration de l'environnement PPO
            env_config = {
                "env": self.env,
                "input_dim": 42,
                "learning_rate": self.config["AI"]["learning_rate"],
                "batch_size": self.config["AI"]["batch_size"],
                "n_epochs": self.config["AI"]["n_epochs"],
                "verbose": 1,
            }

            self.ppo_strategy = PPOStrategy(env_config)

            if self.ppo_strategy.model is None:
                raise ValueError("Échec de l'initialisation du modèle PPO")

            self.ai_enabled = True
            print("✅ Modèles d'IA initialisés avec succès")

        except Exception as e:
            print(f"❌ Erreur initialisation IA: {str(e)}")
            self.ai_enabled = False
            self.dl_model = None
            self.ppo_strategy = None

        # Initialize other components
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.last_telegram_update = datetime.utcnow()
        self.logger = logger

        # Initialisation de l'API Binance
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")

        if self.api_key and self.api_secret:
            self.binance_client = Client(self.api_key, self.api_secret)
            # Modifier cette ligne
            self.binance_connector = (
                BinanceConnector()
            )  # Retirer les paramètres api_key et api_secret
            self.executor = SmartOrderExecutor()
            self.is_live_trading = True
            self.logger.info("Binance API initialized for live trading")
        else:
            self.is_live_trading = False
            self.binance_client = None
            self.binance_connector = None
            self.executor = None
            self.logger.warning(
                "Binance API credentials not found, running in simulation mode"
            )
        print("Vérification des clés API:")
        print(f"API Key présente: {'Oui' if self.api_key else 'Non'}")
        print(f"API Secret présente: {'Oui' if self.api_secret else 'Non'}")

        try:
            print("Configuration de la stratégie PPO...")
            env_config = {
                "env": self.env,
                "input_dim": 42,
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_epochs": 10,
                "verbose": 1,
            }

            # Add error checking for environment
            if not hasattr(self.env, "reset") or not hasattr(self.env, "step"):
                raise ValueError("Trading environment missing required methods")

            self.ppo_strategy = PPOStrategy(env_config)
            if self.ppo_strategy.model is None:
                raise ValueError("PPO model failed to initialize")

            print("✅ PPO Strategy initialized successfully")

        except Exception as e:
            print(f"❌ Erreur initialisation PPO: {str(e)}")
            self.ppo_strategy = None

        # Initialisation de l'analyseur de sentiment
        try:
            self.news_analyzer = NewsSentimentAnalyzer()
            self.news_enabled = True
            self.news_weight = 0.2  # Influence des news dans la décision (20%)
            self.news_update_interval = 300  # 5 minutes
            self.logger.info("News sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize news analyzer: {e}")
            self.news_enabled = False
            self.news_analyzer = None

        print(f"🔄 Bot initialisé avec Telegram: {bool(TELEGRAM_BOT_TOKEN)}")
        print(f"🔄 Trading en direct: {self.is_live_trading}")
        print(f"🔄 IA activée: {self.ai_enabled}")
        print(f"🔄 Analyse de news activée: {self.news_enabled}")

    def check_stop_loss(self, symbol, side):
        """Vérifie si un stop-loss est actif pour le symbole et le côté donnés"""
        try:
            # Si c'est un ordre de vente, pas besoin de vérifier le stop-loss
            if side.upper() == "SELL":
                return False

            # Récupérer les données de marché récentes
            if symbol not in self.market_data:
                return False

            # Calculer la variation de prix récente
            if "1h" in self.market_data[symbol]:
                data = self.market_data[symbol]["1h"]
                if "close" in data and len(data["close"]) >= 2:
                    current_price = data["close"][-1]
                    previous_price = data["close"][-2]
                    price_change = (current_price - previous_price) / previous_price

                    # Si le prix a baissé de plus de 5% récemment, activer le stop-loss
                    if price_change < -0.05:
                        self.logger.warning(
                            f"Stop-loss activé pour {symbol}: variation de {price_change:.2%}"
                        )
                        return True

            return False
        except Exception as e:
            self.logger.error(f"Erreur vérification stop-loss: {e}")
            return False

    async def execute_trade(
        self, symbol, side, amount, price=None, iceberg=False, iceberg_visible_size=0.1
    ):
        """Exécute un ordre de trading avec l'exécuteur intelligent (support natif des ordres iceberg)"""
        if not self.is_live_trading:
            self.logger.info(
                f"SIMULATION: {side} {amount} {symbol} @ {price} (iceberg={iceberg})"
            )
            return {
                "status": "simulated",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "iceberg": iceberg,
            }

        try:
            # Récupération du carnet d'ordres
            bid, ask = await self.binance_connector.get_order_book(symbol)
            orderbook = {"bids": [[float(bid), 1.0]], "asks": [[float(ask), 1.0]]}

            # Récupération des trades récents pour la détection de mouvements
            recent_trades = self.binance_client.get_recent_trades(symbol=symbol)

            # Préparation des données de marché pour l'exécuteur
            market_data = {
                "recent_trades": recent_trades,
                "volatility": self.calculate_volatility(
                    self.market_data.get(symbol, {}).get("1h", {})
                ),
                "regime": self.regime,
            }

            # Exécution de l'ordre avec notre exécuteur intelligent (ajout iceberg)
            result = await self.executor.execute_order(
                symbol=symbol,
                side=side,
                amount=amount,
                orderbook=orderbook,
                market_data=market_data,
                iceberg=iceberg,
                iceberg_visible_size=iceberg_visible_size,
            )

            # Enregistrement du résultat
            if result["status"] == "completed":
                self.logger.info(
                    f"Order executed: {side} {result['filled_amount']} {symbol} @ {result['avg_price']}"
                )
                # Mettre à jour les statistiques
                self._update_performance_metrics(result)

                # Notification Telegram
                iceberg_info = (
                    f"\n🧊 <b>Ordre Iceberg</b> ({result['n_suborders']} sous-ordres)"
                    if result.get("iceberg")
                    else ""
                )
                await self.telegram.send_message(
                    f"💰 <b>Ordre exécuté</b>\n"
                    f"📊 {side} {result['filled_amount']} {symbol} @ {result['avg_price']}\n"
                    f"💵 Total: ${float(result['filled_amount']) * float(result['avg_price']):.2f}"
                    f"{iceberg_info}"
                )

            return result

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error: {e}")
            await self.telegram.send_message(f"⚠️ Erreur API Binance: {e}")
            return {"status": "error", "reason": str(e)}
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            await self.telegram.send_message(f"⚠️ Erreur d'exécution: {e}")
            return {"status": "error", "reason": str(e)}

    def _update_performance_metrics(self, trade_result):
        """Met à jour les métriques de performance après un trade réel"""
        # Chargement des données actuelles
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)

            performance = data["bot_status"]["performance"]

            # Mise à jour des statistiques
            performance["total_trades"] += 1

            # Calcul du profit/perte
            filled_amount = float(trade_result["filled_amount"])
            avg_price = float(trade_result["avg_price"])
            side = trade_result["side"]

            if side() == "buy":
                # Pour un achat, on ne sait pas encore si c'est gagnant
                pass
            elif side() == "sell":
                # Pour une vente, on peut calculer le profit par rapport au prix d'achat moyen
                entry_price = trade_result.get("entry_price", 0)
                if entry_price > 0:
                    profit_pct = (
                        (avg_price / entry_price - 1) * 100
                        if side() == "sell"
                        else (1 - avg_price / entry_price) * 100
                    )
                    profit_amount = filled_amount * avg_price * profit_pct / 100

                    # Mise à jour de la balance
                    performance["balance"] += profit_amount

                    # Mise à jour du win_rate
                    if profit_amount > 0:
                        performance["wins"] = performance.get("wins", 0) + 1
                    else:
                        performance["losses"] = performance.get("losses", 0) + 1

                    performance["win_rate"] = (
                        performance.get("wins", 0) / performance["total_trades"]
                    )

                    # Mise à jour du profit factor
                    performance["total_profit"] = performance.get(
                        "total_profit", 0
                    ) + max(0, profit_amount)
                    performance["total_loss"] = performance.get("total_loss", 0) + max(
                        0, -profit_amount
                    )

                    if performance["total_loss"] > 0:
                        performance["profit_factor"] = (
                            performance["total_profit"] / performance["total_loss"]
                        )

            # Sauvegarde des données mises à jour
            data["bot_status"]["performance"] = performance
            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=4)

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _prepare_features_for_ai(self, symbol):
        """Prépare les features pour les modèles d'IA"""
        try:
            # Récupération des données OHLCV récentes
            ohlcv = self.market_data.get(symbol, {}).get("1h", {})

            if not ohlcv or not isinstance(ohlcv, dict) or "close" not in ohlcv:
                return None

            # Normalisation des données
            closes = np.array(ohlcv.get("close", []))
            highs = np.array(ohlcv.get("high", []))
            lows = np.array(ohlcv.get("low", []))
            volumes = np.array(ohlcv.get("volume", []))

            if len(closes) < 20:
                return None

            # Calcul des indicateurs techniques
            # RSI
            delta = np.diff(closes)
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0.001
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))

            # MACD
            ema12 = np.mean(closes[-12:]) if len(closes) >= 12 else closes[-1]
            ema26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            macd = ema12 - ema26

            # Volatilité
            volatility = (
                np.std(delta[-20:]) / np.mean(closes[-20:]) if len(delta) >= 20 else 0
            )

            # Volume relatif
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            vol_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1

            # Construire le tableau de features normalisées
            features = {
                "close": closes[-20:]
                / closes[-20],  # Normalisation par rapport au premier point
                "high": highs[-20:] / highs[-20],
                "low": lows[-20:] / lows[-20],
                "volume": volumes[-20:] / volumes[-20],
                "rsi": rsi / 100,  # Normalisation entre 0 et 1
                "macd": (macd + 100) / 200,  # Normalisation arbitraire
                "volatility": min(1, volatility * 10),  # Normalisation avec cap à 1
                "vol_ratio": min(1, vol_ratio / 3),  # Normalisation avec cap à 1
            }

            return features

        except Exception as e:
            self.logger.error(f"Error preparing AI features: {e}")
            return None

    async def _merge_signals(self, symbol, dl_prediction, ppo_action):
        """Fusionne les signaux techniques et d'IA"""
        try:
            # Récupération des signaux techniques actuels
            if symbol not in self.market_data:
                self.market_data[symbol] = {}

            if "signals" not in self.market_data[symbol]:
                self.market_data[symbol]["signals"] = {
                    "trend": 0,
                    "momentum": 0,
                    "volatility": 0,
                }

            current_signals = self.market_data[symbol]["signals"]

            # Conversion des prédictions IA en signaux (-1 à 1)
            ai_signal = dl_prediction * 0.7 + ppo_action * 0.3

            # Fusion pondérée des signaux
            for signal_type in current_signals:
                technical_weight = 1 - self.ai_weight
                current_signals[signal_type] = (
                    current_signals[signal_type] * technical_weight
                    + ai_signal * self.ai_weight
                )

            # Mise à jour des signaux
            self.market_data[symbol]["signals"] = current_signals
            self.market_data[symbol]["ai_prediction"] = float(ai_signal)

            return current_signals

        except Exception as e:
            self.logger.error(f"Error merging signals: {e}")
            return {}

    async def _news_analysis_loop(self):
        """Boucle d'analyse des news (version propre sans print/debug)"""
        while True:
            try:
                if not self.news_enabled or not self.news_analyzer:
                    await asyncio.sleep(self.news_update_interval)
                    continue

                self.logger.info("Fetching latest news for sentiment analysis")
                news_data = await self.news_analyzer.fetch_all_news()

                try:
                    sentiment_scores = self.news_analyzer.analyze_sentiment(news_data)
                except Exception:
                    sentiment_scores = []

                try:
                    await self._update_sentiment_data(sentiment_scores)
                except Exception:
                    pass

                try:
                    await self._save_sentiment_data(sentiment_scores, news_data)
                except Exception:
                    pass

                try:
                    await self.telegram.send_news_summary(news_data[:5])
                except Exception:
                    pass

            except Exception as e:
                self.logger.error(f"News analysis error: {e}")

            await asyncio.sleep(self.news_update_interval)

    async def _update_sentiment_data(self, sentiment_scores):
        """Met à jour les données de marché avec le sentiment"""
        for item in sentiment_scores:
            symbol = item.get("symbol", "")
            score = item.get("sentiment", 0)
            if symbol and symbol in self.market_data:
                # Mise à jour des données de marché avec le sentiment
                self.market_data[symbol]["sentiment"] = score
                self.market_data[symbol]["sentiment_timestamp"] = time.time()

                # Ajustement des signaux en fonction du sentiment
                if "signals" in self.market_data[symbol]:
                    signals = self.market_data[symbol]["signals"]

                    # Le sentiment influence tous les signaux
                    for signal_type in signals:
                        # Ajustement proportionnel au sentiment avec pondération du temps
                        time_factor = 1.0  # Diminue avec le temps
                        if "sentiment_timestamp" in self.market_data[symbol]:
                            elapsed_time = (
                                time.time()
                                - self.market_data[symbol]["sentiment_timestamp"]
                            )
                            time_factor = max(
                                0.2, 1.0 - (elapsed_time / (3600 * 12))
                            )  # Décroît sur 12h

                        # Ajuster le poids du sentiment en fonction de son intensité
                        sentiment_weight = self.news_weight * (1 + abs(score))

                        # Donner plus d'importance aux nouvelles très positives ou très négatives
                        if abs(score) > 0.7:
                            sentiment_weight *= 1.5

                        sentiment_adjustment = score * sentiment_weight * time_factor
                        signals[signal_type] += sentiment_adjustment

    async def _save_sentiment_data(self, sentiment_scores, news_data):
        """Sauvegarde les données de sentiment pour l'interface"""
        # Formatage des données pour l'interface
        headlines = []
        if isinstance(news_data, list):
            for item in news_data[:10]:
                if isinstance(item, dict) and "title" in item:
                    headlines.append(item["title"])

        sentiment_data = {
            "timestamp": datetime.now().isoformat(),
            "scores": sentiment_scores,
            "latest_news": headlines,
        }

        # Mise à jour du fichier shared_data.json
        try:
            with open(self.data_file, "r") as f:
                shared_data = json.load(f)

            shared_data["sentiment"] = sentiment_data

            with open(self.data_file, "w") as f:
                json.dump(shared_data, f, indent=4)

        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")

    async def generate_market_analysis_report(self):
        """Génère un rapport d'analyse de marché détaillé"""
        debug_market_data_structure(
            self.market_data, self.pairs_valid, ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
        report = (
            f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {get_current_time()}\n"
            f"Current User's Login: {CURRENT_USER}\n"
            "╔═════════════════════════════════════════════════╗\n"
            "║           RAPPORT D'ANALYSE DE MARCHÉ           ║\n"
            "╠═════════════════════════════════════════════════╣\n"
            f"║ Régime: {self.regime}                               ║\n"
            "╚═════════════════════════════════════════════════╝\n\n"
            "    📊 Analyse par Timeframe/Paire :\n"
        )
        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                if pair_key not in self.market_data or tf not in self.market_data.get(
                    pair_key, {}
                ):
                    print(f"ABSENT: {pair_key} {tf}")

        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        for tf in timeframes:
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                report += f"""
    🕒 {tf} | {pair} :
    ├─ 📈 Tendance: {self.get_trend_analysis(pair, tf)}
    ├─ 📊 Volatilité: {self.get_volatility_analysis(pair, tf)}
    ├─ 📉 Volume: {self.get_volume_analysis(pair, tf)}
    └─ 🎯 Signal dominant: {self.get_dominant_signal(pair, tf)}
    """

        # Ajout des informations d'IA si disponibles
        if self.ai_enabled:
            report += "\n    🧠 Analyse IA :\n"
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if (
                    pair_key in self.market_data
                    and "ai_prediction" in self.market_data[pair_key]
                ):
                    ai_score = self.market_data[pair_key]["ai_prediction"]
                    ai_signal = (
                        "ACHAT"
                        if ai_score > 0.6
                        else "VENTE" if ai_score < 0.4 else "NEUTRE"
                    )
                    report += f"""
    🤖 {pair} :
    └─ Prédiction: {ai_signal} ({ai_score:.2f})
    """

        # Ajout des informations de sentiment si disponibles
        if self.news_enabled:
            report += "\n    📰 Analyse de Sentiment :\n"
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if (
                    pair_key in self.market_data
                    and "sentiment" in self.market_data[pair_key]
                ):
                    sentiment_score = self.market_data[pair_key]["sentiment"]
                    sentiment_type = (
                        "Positif"
                        if sentiment_score > 0.2
                        else "Négatif" if sentiment_score < -0.2 else "Neutre"
                    )
                    report += f"""
    📊 {pair} :
    └─ Sentiment: {sentiment_type} ({sentiment_score:.2f})
    """

        return report

    def calculate_trend(self, data):
        try:
            if isinstance(data, dict) and "close" in data:
                closes = data["close"][-20:]
                if not closes or len(closes) < 10:
                    return 0.0
                ma_fast = sum(closes[-10:]) / 10
                ma_slow = sum(closes) / len(closes)
                trend = (ma_fast / ma_slow) - 1
                return float(trend)
            return 0.0
        except Exception as e:
            print("DEBUG calculate_trend error:", e)
            return 0.0

    def calculate_volatility(self, data):
        try:
            if isinstance(data, dict) and "close" in data:
                closes = data["close"][-20:]
                if not closes or len(closes) < 2:
                    return 0.0
                returns = np.diff(np.log(closes))
                return float(np.std(returns) * np.sqrt(252))
            return 0.0
        except Exception as e:
            return 0.0

    def calculate_volume_profile(self, data):
        try:
            if isinstance(data, dict) and "volume" in data:
                volumes = data["volume"][-20:]
                if not volumes or len(volumes) < 2:
                    return 1.0
                current_vol = volumes[-1]
                avg_vol = sum(volumes) / len(volumes)
                return float(current_vol / avg_vol) if avg_vol > 0 else 1.0
            return 1.0
        except Exception as e:
            print("DEBUG calculate_volume_profile error:", e)
        return 1.0

    def get_trend_analysis(self, pair, timeframe):
        try:
            pair_key = pair.replace("/", "").upper()
            if pair_key in self.market_data and timeframe in self.market_data[pair_key]:
                trend = self.calculate_trend(self.market_data[pair_key][timeframe])
                if trend > 0.02:
                    return "Haussière"
                elif trend < -0.02:
                    return "Baissière"
                return "Neutre"
            return "N/A"
        except Exception as e:
            return "N/A"

    def get_volatility_analysis(self, pair, timeframe):
        """Analyse de volatilité détaillée"""
        try:
            pair_key = pair.replace("/", "").upper()
            if pair_key in self.market_data and timeframe in self.market_data[pair_key]:
                vol = self.calculate_volatility(self.market_data[pair_key][timeframe])
                if vol > 0.8:
                    return "Élevée"
                elif vol > 0.4:
                    return "Moyenne"
                return "Faible"
            return "N/A"
        except Exception as e:
            print(f"DEBUG get_volatility_analysis error: {e}")
            return "N/A"

    def get_volume_analysis(self, pair, timeframe):
        """Analyse du volume"""
        try:
            pair_key = pair.replace("/", "").upper()
            if pair_key in self.market_data and timeframe in self.market_data[pair_key]:
                data = self.market_data[pair_key][timeframe]
                if (
                    data
                    and "volume" in data
                    and isinstance(data["volume"], list)
                    and len(data["volume"]) >= 2
                ):
                    vol = self.calculate_volume_profile(data)
                    if vol > 1.5:
                        return "Fort"
                    elif vol > 0.7:
                        return "Moyen"
                    return "Faible"
                else:
                    return "N/A"
            return "N/A"
        except Exception as e:
            print(f"DEBUG get_volume_analysis error: {e}")
            return "N/A"

    def get_dominant_signal(self, pair, timeframe):
        """Signal dominant"""
        try:
            trend = self.get_trend_analysis(pair, timeframe)
            vol = self.get_volatility_analysis(pair, timeframe)
            volume = self.get_volume_analysis(pair, timeframe)
            if trend == "Haussière" and vol != "Élevée" and volume != "Faible":
                return "LONG"
            elif trend == "Baissière" and vol != "Élevée" and volume != "Faible":
                return "SHORT"
            elif vol == "Élevée" or volume == "Faible":
                return "ATTENTE"
            return "NEUTRE"
        except Exception as e:
            print(f"DEBUG get_dominant_signal error: {e}")
            return "N/A"

    async def study_market(self, timeframe):
        """Analyse le marché"""
        try:
            await asyncio.sleep(0.5)  # Simule le temps de calcul

            # Récupération des données de marché
            if self.is_live_trading:
                # Utilisation de l'API Binance pour les données réelles
                await self._fetch_real_market_data()
            else:
                # Utilisation de données simulées
                self.market_data = await self.get_latest_data()

            # Analyse du régime global
            volatility = self.calculate_volatility(
                self.market_data.get("BTCUSDT", {}).get("1h", {})
            )
            trend = self.calculate_trend(
                self.market_data.get("BTCUSDT", {}).get("1h", {})
            )

            if volatility > 0.8:
                self.regime = MARKET_REGIMES["VOLATILE"]
            elif trend > 0.02:
                self.regime = MARKET_REGIMES["TRENDING_UP"]
            elif trend < -0.02:
                self.regime = MARKET_REGIMES["TRENDING_DOWN"]
            else:
                self.regime = MARKET_REGIMES["RANGING"]

            # Si l'IA est activée, ajoutez les prédictions de l'IA
            if self.ai_enabled:
                await self._add_ai_predictions()

            return self.regime, self.market_data, {}
        except Exception as e:
            self.logger.error(f"Erreur analyse marché: {e}")
            return self.regime, None, {}

    async def _fetch_real_market_data(self):
        """Récupère les données de marché réelles depuis Binance"""
        try:
            if not self.is_live_trading or not self.binance_client:
                return

            timeframes = {
                "1m": Client.KLINE_INTERVAL_1MINUTE,
                "5m": Client.KLINE_INTERVAL_5MINUTE,
                "15m": Client.KLINE_INTERVAL_15MINUTE,
                "1h": Client.KLINE_INTERVAL_1HOUR,
                "4h": Client.KLINE_INTERVAL_4HOUR,
                "1d": Client.KLINE_INTERVAL_1DAY,
            }

            market_data = {}

            for pair in self.pairs_valid:
                pair_binance = pair.replace("/", "")
                market_data[pair_binance] = {}

                for tf_name, tf_value in timeframes.items():
                    try:
                        # Récupération des données historiques
                        klines = self.binance_client.get_klines(
                            symbol=pair_binance, interval=tf_value, limit=100
                        )

                        # Conversion au format OHLCV
                        ohlcv = {
                            "open": [float(k[1]) for k in klines],
                            "high": [float(k[2]) for k in klines],
                            "low": [float(k[3]) for k in klines],
                            "close": [float(k[4]) for k in klines],
                            "volume": [float(k[5]) for k in klines],
                            "timestamp": [int(k[0]) for k in klines],
                        }

                        market_data[pair_binance][tf_name] = ohlcv

                    except BinanceAPIException as e:
                        self.logger.error(
                            f"Binance API error for {pair} {tf_name}: {e}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error fetching data for {pair} {tf_name}: {e}"
                        )

            self.market_data = market_data

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")

    async def _add_ai_predictions(self):
        """Ajoute les prédictions des modèles d'IA aux données de marché"""
        if not self.ai_enabled or not self.dl_model or not self.ppo_strategy:
            return

        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()

            # Préparation des features pour l'IA
            features = await self._prepare_features_for_ai(pair_key)

            if features is not None:
                try:
                    # Prédiction du modèle CNN-LSTM
                    dl_prediction = self.dl_model.predict(features)

                    # Recommandation du modèle PPO
                    ppo_action = self.ppo_strategy.get_action(features)

                    # Fusion des signaux IA avec les signaux techniques
                    await self._merge_signals(pair_key, dl_prediction, ppo_action)

                except Exception as e:
                    self.logger.error(f"Error getting AI predictions for {pair}: {e}")

    async def fetch_news(self):
        """Récupère les news de différentes sources"""
        try:
            news = []
            # Essayer plusieurs sources
            try:
                coindesk_news = await self.fetch_coindesk_news()
                news.extend(coindesk_news)
            except Exception as e:
                self.logger.warning(f"Erreur CoinDesk: {e}")

            try:
                cointelegraph_news = await self.fetch_cointelegraph_news()
                news.extend(cointelegraph_news)
            except Exception as e:
                self.logger.warning(f"Erreur Cointelegraph: {e}")

            return news
        except Exception as e:
            self.logger.error(f"Erreur générale news: {e}")
            return []

    async def study_market_period(self, symbol, start_time, end_time, timeframe="1h"):
        """Étudie le marché sur une période définie et établit un plan de trading"""
        try:
            # Convertir les dates en timestamps (ms)
            start_ts = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000)

            # Récupérer les données historiques
            tf_binance = getattr(Client, f"KLINE_INTERVAL_{timeframe.upper()}")
            klines = self.binance_client.get_historical_klines(
                symbol=symbol, interval=tf_binance, start_str=start_ts, end_str=end_ts
            )

            # Convertir en DataFrame
            ohlcv = {
                "open": [float(k[1]) for k in klines],
                "high": [float(k[2]) for k in klines],
                "low": [float(k[3]) for k in klines],
                "close": [float(k[4]) for k in klines],
                "volume": [float(k[5]) for k in klines],
                "timestamp": [int(k[0]) for k in klines],
            }

            # Analyser les données
            trend = self.calculate_trend(ohlcv)
            volatility = self.calculate_volatility(ohlcv)
            volume_profile = self.calculate_volume_profile(ohlcv)

            # Identifier les régimes de marché
            if volatility > 0.8:
                regime = MARKET_REGIMES["VOLATILE"]
                strategy = "Protection du capital - trades limités, stop-loss étroits"
            elif trend > 0.02:
                regime = MARKET_REGIMES["TRENDING_UP"]
                strategy = "Suivre la tendance - positions longues, trailing stop"
            elif trend < -0.02:
                regime = MARKET_REGIMES["TRENDING_DOWN"]
                strategy = "Ventes courtes ou attente - protection des positions"
            else:
                regime = MARKET_REGIMES["RANGING"]
                strategy = "Range trading - achats aux supports, ventes aux résistances"

            # Préparer le rapport d'analyse
            analysis_report = {
                "symbol": symbol,
                "period": f"{start_time} à {end_time}",
                "timeframe": timeframe,
                "data_points": len(klines),
                "regime": regime,
                "trend": trend,
                "volatility": volatility,
                "volume_profile": volume_profile,
                "strategy": strategy,
                "key_levels": self._identify_key_levels(ohlcv),
            }

            # Envoyer le rapport sur Telegram
            report_message = (
                f"📊 <b>Analyse de Marché: {symbol}</b>\n\n"
                f"⏱️ Période: {start_time} à {end_time}\n"
                f"📈 Régime: {regime}\n"
                f"🔍 Tendance: {trend:.2%}\n"
                f"📏 Volatilité: {volatility:.2%}\n\n"
                f"🎯 <b>Stratégie recommandée:</b>\n{strategy}\n\n"
                f"🔑 <b>Niveaux clés:</b>\n"
            )

            for level in analysis_report["key_levels"][:3]:
                report_message += f"- {level['type']}: {level['price']:.2f}\n"

            await self.telegram.send_message(report_message)

            return analysis_report

        except Exception as e:
            self.logger.error(f"Error studying market period: {e}")
            return None

    def _identify_key_levels(self, ohlcv):
        """Identifie les niveaux clés (support/résistance) dans les données"""
        levels = []

        try:
            highs = np.array(ohlcv["high"])
            lows = np.array(ohlcv["low"])
            closes = np.array(ohlcv["close"])

            # Identifier les sommets locaux (résistances potentielles)
            for i in range(2, len(highs) - 2):
                if (
                    highs[i] > highs[i - 1]
                    and highs[i] > highs[i - 2]
                    and highs[i] > highs[i + 1]
                    and highs[i] > highs[i + 2]
                ):
                    levels.append(
                        {"price": highs[i], "type": "Résistance", "strength": 1}
                    )

            # Identifier les creux locaux (supports potentiels)
            for i in range(2, len(lows) - 2):
                if (
                    lows[i] < lows[i - 1]
                    and lows[i] < lows[i - 2]
                    and lows[i] < lows[i + 1]
                    and lows[i] < lows[i + 2]
                ):
                    levels.append({"price": lows[i], "type": "Support", "strength": 1})

            # Regrouper les niveaux proches
            grouped_levels = []
            sorted_levels = sorted(levels, key=lambda x: x["price"])

            if sorted_levels:
                current_group = [sorted_levels[0]]
                current_price = sorted_levels[0]["price"]

                for level in sorted_levels[1:]:
                    # Si le niveau est proche du groupe actuel (0.5% de différence)
                    if abs(level["price"] - current_price) / current_price < 0.005:
                        current_group.append(level)
                    else:
                        # Calculer le niveau moyen du groupe
                        avg_price = sum(l["price"] for l in current_group) / len(
                            current_group
                        )
                        avg_strength = sum(l["strength"] for l in current_group)
                        type_counts = {"Support": 0, "Résistance": 0}
                        for l in current_group:
                            type_counts[l["type"]] += 1

                        # Déterminer le type dominant
                        level_type = (
                            "Support"
                            if type_counts["Support"] > type_counts["Résistance"]
                            else "Résistance"
                        )

                        grouped_levels.append(
                            {
                                "price": avg_price,
                                "type": level_type,
                                "strength": avg_strength,
                            }
                        )

                        # Commencer un nouveau groupe
                        current_group = [level]
                        current_price = level["price"]

                # Ajouter le dernier groupe
                if current_group:
                    avg_price = sum(l["price"] for l in current_group) / len(
                        current_group
                    )
                    avg_strength = sum(l["strength"] for l in current_group)
                    type_counts = {"Support": 0, "Résistance": 0}
                    for l in current_group:
                        type_counts[l["type"]] += 1

                    level_type = (
                        "Support"
                        if type_counts["Support"] > type_counts["Résistance"]
                        else "Résistance"
                    )

                    grouped_levels.append(
                        {
                            "price": avg_price,
                            "type": level_type,
                            "strength": avg_strength,
                        }
                    )

            # Trier par force décroissante
            return sorted(grouped_levels, key=lambda x: x["strength"], reverse=True)

        except Exception as e:
            self.logger.error(f"Error identifying key levels: {e}")
            return []

    async def send_telegram_updates(self):
        performance = self.get_performance_metrics()
        if not performance:
            performance = {
                "balance": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
            }
        await self.telegram.send_performance_update(performance)
        report = await self.generate_market_analysis_report()
        await self.telegram.send_message(report)

    def initialize_shared_data(self):
        """Initialise le fichier de données partagées"""
        data = {
            "timestamp": get_current_time(),
            "user": CURRENT_USER,
            "bot_status": {
                "regime": self.regime,
                "cycle": self.current_cycle,
                "last_update": get_current_time(),
                "performance": {
                    "total_trades": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "balance": 10000,
                    "wins": 0,
                    "losses": 0,
                    "total_profit": 0,
                    "total_loss": 0,
                },
            },
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=4)

    def save_shared_data(self):
        """Met à jour les données partagées"""
        data = {
            "timestamp": get_current_time(),
            "user": CURRENT_USER,
            "bot_status": {
                "regime": self.regime,
                "cycle": self.current_cycle,
                "last_update": get_current_time(),
                "performance": self.get_performance_metrics(),
            },
            "market_data": self.market_data,
            "indicators": self.indicators,  # <--- AJOUT DÈS LA CRÉATION
        }

        # Ajoute les prédictions IA si besoin
        if self.ai_enabled:
            ai_predictions = {}
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if (
                    pair_key in self.market_data
                    and "ai_prediction" in self.market_data[pair_key]
                ):
                    ai_predictions[pair] = self.market_data[pair_key]["ai_prediction"]
            data["ai_predictions"] = ai_predictions

        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=4)

    def get_performance_metrics(self):
        """Récupère les métriques de performance actuelles"""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)

            # AJOUT ICI : récupération du solde réel Binance
            real_balance = self.get_binance_real_balance("USDC")
            if real_balance is not None:
                data["bot_status"]["performance"]["balance"] = real_balance

            return data["bot_status"]["performance"]
        except:
            # Retourne des métriques simulées si le fichier n'existe pas
            return {
                "total_trades": self.current_cycle * 2,
                "win_rate": 0.62 + (self.current_cycle * 0.001),
                "profit_factor": 1.85 + (self.current_cycle * 0.01),
                "balance": 10000 + (self.current_cycle * 100),
                "wins": int(self.current_cycle * 1.2),
                "losses": self.current_cycle - int(self.current_cycle * 1.2),
                "total_profit": self.current_cycle * 150,
                "total_loss": self.current_cycle * 50,
            }

    async def _setup_components(self):
        try:
            # >>>> DEMARRAGE WS <<<<
            await self.ws_collector.start()
            # >>>> FIN AJOUT <<<<

            # Lancement du processus d'analyse des news
            if self.news_enabled and self.news_analyzer:
                asyncio.create_task(self._news_analysis_loop())
                self.logger.info("News analysis loop started")

                # Lancement du processus d'analyse des news
                if self.news_enabled and self.news_analyzer:
                    asyncio.create_task(self._news_analysis_loop())
                    self.logger.info("News analysis loop started")

                # Initialisation des connexions WebSocket Binance si en mode trading réel
                if self.is_live_trading:
                    # Ici vous pouvez initialiser les connexions WebSocket
                    self.logger.info("Binance WebSocket connections initialized")

                await asyncio.sleep(0.5)  # Simule le temps de configuration
                return True

        except Exception as e:
            self.logger.error(f"Error setting up components: {e}")
            return False

    def choose_strategy(self, regime, indicators):
        """Choisit la stratégie"""
        return f"{regime}"

    async def get_latest_data(self):
        """Récupère les dernières données simulées"""
        await asyncio.sleep(0.3)  # Simule le temps de récupération

        # Données simulées pour toutes les paires configurées
        data = {}
        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            data[pair_key] = {}

            # Génération de données OHLCV pour différents timeframes
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                base_price = 100 if "BTC" in pair else 1.5
                volatility = (
                    0.01 if tf in ["1m", "5m"] else 0.02 if tf == "15m" else 0.05
                )

                # Génération de données avec une petite tendance aléatoire
                n_points = 100
                trend = np.random.choice([-0.0001, 0.0001]) * np.arange(n_points)
                noise = np.random.normal(0, volatility, n_points)
                price_movement = trend + noise

                # Création des séries de prix
                closes = base_price * (1 + np.cumsum(price_movement))
                opens = closes * (1 + np.random.normal(0, 0.001, n_points))
                highs = np.maximum(opens, closes) * (
                    1 + np.abs(np.random.normal(0, 0.003, n_points))
                )
                lows = np.minimum(opens, closes) * (
                    1 - np.abs(np.random.normal(0, 0.003, n_points))
                )
                volumes = np.random.normal(1000, 200, n_points)

                data[pair_key][tf] = {
                    "open": opens.tolist(),
                    "high": highs.tolist(),
                    "low": lows.tolist(),
                    "close": closes.tolist(),
                    "volume": volumes.tolist(),
                    "timestamp": [
                        int(datetime.now().timestamp()) - i * 60
                        for i in range(n_points)
                    ],
                }

                # Ajout des signaux simulés
                if "signals" not in data[pair_key]:
                    data[pair_key]["signals"] = {
                        "trend": np.random.uniform(-0.5, 0.5),
                        "momentum": np.random.uniform(-0.5, 0.5),
                        "volatility": np.random.uniform(0, 1),
                    }

        return data

    def add_indicators(self, df):
        """
        Calcule TOUS les indicateurs possibles (130+) avec pandas-ta ou ta.
        Retourne un dictionnaire {nom_indicateur: dernière_valeur}
        """
        import pandas as pd

        try:
            print(
                f"[DEBUG add_indicators] Entrée type: {type(df)}, shape: {getattr(df, 'shape', 'N/A')}"
            )
            # 1. Gestion entrée : DataFrame, liste de dicts, liste de listes
            if isinstance(df, list):
                if len(df) == 0:
                    self.logger.error("add_indicators: Liste reçue vide")
                    print("[DEBUG add_indicators] Liste reçue vide")
                    return None
                if isinstance(df[0], dict):
                    df = pd.DataFrame(df)
                elif isinstance(df[0], (list, tuple)):
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]
                    df = pd.DataFrame(df, columns=columns)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                else:
                    self.logger.error(
                        "add_indicators: Format de liste non pris en charge"
                    )
                    print("[DEBUG add_indicators] Format de liste non pris en charge")
                    return None
            if not isinstance(df, pd.DataFrame):
                self.logger.error("add_indicators: df n'est pas un DataFrame")
                print("[DEBUG add_indicators] df n'est pas un DataFrame")
                return None

            required_cols = {"open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                self.logger.error(
                    f"add_indicators: Colonnes manquantes: {required_cols - set(df.columns)} | Colonnes actuelles: {df.columns.tolist()}"
                )
                print(
                    f"[DEBUG add_indicators] Colonnes manquantes: {required_cols - set(df.columns)}"
                )
                return None

            # 2. Check taille minimale du DataFrame pour éviter erreurs ta-lib/pandas-ta
            MIN_LEN = 30  # ou 50 pour plus de sécurité
            if df is None or len(df) < MIN_LEN:
                self.logger.warning(
                    f"DataFrame vide ou insuffisant ({0 if df is None else len(df)}) lignes"
                )
                print(
                    f"[DEBUG add_indicators] DataFrame trop court ({0 if df is None else len(df)}) lignes"
                )
                return None

            # 3. Sécurité : trier par timestamp pour tous les indicateurs (VWAP, etc.)
            if "timestamp" in df.columns:
                print(
                    f"[DEBUG add_indicators] timestamp dtype: {df['timestamp'].dtype}"
                )
                df = df.sort_values("timestamp")
                df = df.drop_duplicates(subset="timestamp", keep="last")
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
                df = df[~df.index.duplicated(keep="last")]

            if df.empty:
                self.logger.warning(
                    "DataFrame vide, impossible de calculer les indicateurs"
                )
                print("[DEBUG add_indicators] DataFrame vide après tri/formatage")
                return None

            # DEBUG: Affiche la taille du DataFrame
            print(f"[DEBUG add_indicators] Calcul sur DF {df.shape}")

            # 4. Calcul de TOUS les indicateurs
            indicators = {}
            base_cols = ["timestamp", "open", "high", "low", "close", "volume"]

            try:
                import pandas_ta as ta

                if hasattr(df, "ta") and hasattr(df.ta, "strategy"):
                    df_ta = df.copy()
                    df_ta.ta.strategy("All")
                    indicators = {
                        col: df_ta[col].iloc[-1]
                        for col in df_ta.columns
                        if col not in base_cols
                    }
                else:
                    raise ImportError("pandas_ta n'est pas disponible correctement")
            except Exception as e:
                self.logger.warning(
                    f"pandas_ta indisponible ou erreur ({e}), tentative avec ta..."
                )
                print(
                    f"[DEBUG add_indicators] pandas_ta indisponible ou erreur ({e}), tentative avec ta..."
                )
                try:
                    import ta

                    df_reset = df.copy().reset_index()
                    df_with_indicators = ta.add_all_ta_features(
                        df_reset,
                        open="open",
                        high="high",
                        low="low",
                        close="close",
                        volume="volume",
                        fillna=True,
                    )
                    indicators = {
                        col: df_with_indicators[col].iloc[-1]
                        for col in df_with_indicators.columns
                        if col not in base_cols
                    }
                except ImportError:
                    self.logger.error("Ni pandas_ta ni ta n'est installé !")
                    print("[DEBUG add_indicators] Ni pandas_ta ni ta n'est installé !")
                    return None
                except Exception as e2:
                    self.logger.error(f"Erreur ta.add_all_ta_features : {e2}")
                    print(
                        f"[DEBUG add_indicators] Exception ta.add_all_ta_features: {e2}"
                    )
                    return None

            self.logger.info(
                f"✅ {len(indicators)} indicateurs extraits automatiquement sur {df.shape[0]} lignes"
            )
            print(
                f"[DEBUG add_indicators] {len(indicators)} indicateurs extraits: {list(indicators.keys())[:5]}"
            )
            return indicators

        except Exception as e:
            self.logger.error(f"❌ Erreur calcul indicateurs: {e}")
            print(f"[DEBUG add_indicators] Exception: {e}")
            return None

    async def analyze_signals(self, df, indicators):
        action = "neutral"
        confidence = 0.5

        try:
            if indicators:
                # Signaux plus sensibles
                trend_signal = 0
                if indicators.get("sma_20", 0) > indicators.get("sma_50", 0):
                    trend_signal += 0.4
                else:
                    trend_signal -= 0.4

                momentum_signal = 0
                rsi = indicators.get("rsi", 50)
                if rsi > 70:
                    momentum_signal -= 0.6
                elif rsi < 30:
                    momentum_signal += 0.6

                macd = indicators.get("macd", 0)
                macd_signal = indicators.get("macd_signal", 0)
                macd_strength = abs(macd - macd_signal) / max(abs(macd), 0.01)
                if macd > macd_signal:
                    momentum_signal += 0.4 * macd_strength
                else:
                    momentum_signal -= 0.4 * macd_strength

                combined_signal = (trend_signal + momentum_signal) / 2

                if combined_signal > 0.15:  # Seuil plus bas
                    action = "buy"
                    confidence = 0.5 + min(0.5, abs(combined_signal))
                elif combined_signal < -0.15:  # Seuil plus bas
                    action = "sell"
                    confidence = 0.5 + min(0.5, abs(combined_signal))

        except Exception as e:
            self.logger.error(f"Error analyzing signals: {e}")

        return {"action": action, "confidence": confidence}


def load_config():
    """Charge la configuration"""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            return config.get("valid_pairs", ["BTC/USDT", "ETH/USDT"])
    except:
        return ["BTC/USDT", "ETH/USDT"]


async def run_clean_bot():
    """
    Fonction principale du bot de trading
    Gère l'initialisation, l'analyse de marché et l'exécution des stratégies
    """
    print(">>> RUN_CLEAN_BOT DEMARRE <<<")
    orderflow_indicators = AdvancedIndicators()
    logger = logging.getLogger(__name__)

    async def initialize_bot():
        """Initialisation du bot et de ses composants"""
        print(">>> INITIALIZE_BOT <<<")
        try:
            print("\n=== DÉMARRAGE DU BOT ===")
            print("🚀 Trading Bot Ultimate v4 - Version Ultra-Propre")

            # Configuration initiale
            valid_pairs = load_config()

            # Création et configuration du bot
            bot = TradingBotM4()
            bot.pairs_valid = valid_pairs

            if not await bot._setup_components():
                raise Exception("Échec de l'initialisation des composants")

            if bot.is_live_trading:
                await bot._fetch_real_market_data()
                for sym in bot.market_data:
                    print(f"{sym}: {list(bot.market_data[sym].keys())}")

            # Rapport initial
            initial_report = await bot.generate_market_analysis_report()

            await bot.telegram.send_message(
                "🚀 <b>Bot Trading démarré</b>\n"
                "✅ Initialisation réussie\n"
                f"📊 Paires configurées: {', '.join(valid_pairs)}\n\n"
                f"{initial_report}"
            )

            print("✅ Bot initialized successfully")
            return bot, valid_pairs

        except Exception as e:
            logger.error(f"Erreur d'initialisation: {e}")
            raise

    async def market_analysis_cycle(bot, pair, market_data):
        try:
            pair_key = pair.replace("/", "")
            if not market_data or pair_key not in market_data:
                return None

            # Ici :
            ohlcv_df = bot.ws_collector.get_dataframe(pair_key, "1h")
            if ohlcv_df is None or len(ohlcv_df) < 20:
                return None

            indicators_data = bot.add_indicators(ohlcv_df)
            signal = await bot.analyze_signals(ohlcv_df, indicators_data)
            # etc...
        except Exception as e:
            logger.error(f"Erreur analyse {pair}: {e}")
            return None

    async def execute_trading_cycle(bot, valid_pairs):
        """Exécute un cycle complet de trading"""
        try:
            # 0. Import avancé des indicateurs orderflow (à placer en haut du fichier !)
            try:
                from src.analysis.technical.advanced.advanced_indicators import (
                    AdvancedIndicators,
                )

                orderflow_indicators = AdvancedIndicators()
            except Exception as e:
                orderflow_indicators = None
                print("[Orderflow] Impossible d'importer AdvancedIndicators:", e)

            # 1. Injection des données live WS dans market_data (remplace le fetch market_data historique !)
            for pair in bot.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if pair_key not in bot.market_data:
                    bot.market_data[pair_key] = {}
                for tf in bot.config["TRADING"]["timeframes"]:
                    df = bot.ws_collector.get_dataframe(pair_key, tf)
                    print(
                        f"[DEBUG CYCLE] DF {pair_key} {tf} : {len(df) if df is not None else 'None'} lignes, colonnes: {list(df.columns) if df is not None else 'None'}"
                    )
                    if df is not None and not df.empty:
                        bot.market_data[pair_key][tf] = {
                            "open": df["open"].tolist(),
                            "high": df["high"].tolist(),
                            "low": df["low"].tolist(),
                            "close": df["close"].tolist(),
                            "volume": df["volume"].tolist(),
                            "timestamp": [
                                int(pd.Timestamp(t).timestamp())
                                for t in df["timestamp"]
                            ],
                        }
                        # 1bis. Calcul et injection des indicateurs orderflow avancés
                        if orderflow_indicators is not None:
                            try:
                                bid_ask = None
                                liquidity_wave = None
                                smart_money = None
                                if hasattr(orderflow_indicators, "_bid_ask_ratio"):
                                    bid_ask = orderflow_indicators._bid_ask_ratio(df)
                                if hasattr(orderflow_indicators, "_liquidity_wave"):
                                    liquidity_wave = (
                                        orderflow_indicators._liquidity_wave(df)
                                    )
                                if hasattr(orderflow_indicators, "_smart_money_index"):
                                    smart_money = (
                                        orderflow_indicators._smart_money_index(df)
                                    )
                                bot.market_data[pair_key][tf]["orderflow"] = {
                                    "bid_ask_ratio": bid_ask,
                                    "liquidity_wave": liquidity_wave,
                                    "smart_money_index": smart_money,
                                }
                            except Exception as e:
                                print(f"[Orderflow] Erreur calcul {pair_key} {tf}: {e}")
                    else:
                        print(
                            f"[DEBUG CYCLE] DataFrame vide ou None pour {pair_key} {tf}"
                        )

            for sym in bot.market_data:
                print(f"[DEBUG MARKET_DATA] {sym}: {list(bot.market_data[sym].keys())}")

            # 2. Analyse de marché
            regime, market_data, indicators = await bot.study_market("7d")
            strategy = bot.choose_strategy(regime, indicators)
            print(f"🎯 Stratégie active: {strategy}")

            # 3. Détection d'arbitrage
            await handle_arbitrage_opportunities(bot)

            # 4. Analyse des paires
            trade_decisions = []
            for pair in valid_pairs:
                decision = await market_analysis_cycle(bot, pair, bot.market_data)
                if decision:
                    trade_decisions.append(decision)

            # 5. Exécution des trades
            await execute_trade_decisions(bot, trade_decisions)

            return trade_decisions, regime

        except Exception as e:
            print(f"[DEBUG] Exception in execute_trading_cycle: {e}")
            logger.error(f"Erreur cycle trading: {e}")
            raise

    # Fonction principale
    async def main():
        try:
            # Initialisation
            bot, valid_pairs = await initialize_bot()

            # Analyse initiale du marché
            regime, _, _ = await bot.study_market("7d")
            print(f"🔈 Régime de marché détecté: {regime}")

            # Boucle principale
            cycle = 0
            while True:
                cycle += 1
                start = datetime.utcnow()
                try:
                    print(f"\n🔄 Cycle {cycle} - {start.strftime('%H:%M:%S')}")

                    # Exécution du cycle de trading
                    trade_decisions, regime = await execute_trading_cycle(
                        bot, valid_pairs
                    )

                    # Mise à jour des données
                    bot.current_cycle = cycle
                    bot.regime = regime

                    # Pour chaque paire et timeframe, calcule et stocke les indicateurs
                    bot.indicators = {}
                    print(f"[DEBUG] PAIRS VALID: {bot.pairs_valid}")
                    for pair in bot.pairs_valid:
                        pair_key = pair.replace("/", "").upper()
                        for tf in bot.config["TRADING"]["timeframes"]:
                            df = bot.ws_collector.get_dataframe(pair_key, tf)
                            print(
                                f"[DEBUG] DF {pair_key} {tf} : {len(df) if df is not None else 'None'} lignes, colonnes: {list(df.columns) if df is not None else 'None'}"
                            )
                            if df is not None and not df.empty:
                                try:
                                    indics = bot.add_indicators(df)
                                    print(
                                        f"[DEBUG] add_indicators result pour {pair_key} {tf}: {type(indics)} ({'OK' if indics else 'None/Empty'})"
                                    )
                                    if indics is not None:
                                        print(
                                            f"[DEBUG] Nb indicateurs extraits pour {pair_key} {tf}: {len(indics)} | Exemples: {list(indics.keys())[:5]}"
                                        )
                                    if pair_key not in bot.indicators:
                                        bot.indicators[pair_key] = {}
                                    bot.indicators[pair_key][tf] = indics
                                except Exception as e:
                                    print(
                                        f"[DEBUG] Exception add_indicators {pair_key} {tf}: {e}"
                                    )
                                    bot.logger.error(
                                        f"Error calculating indicators for {pair_key} {tf}: {e}"
                                    )
                            else:
                                print(
                                    f"[DEBUG] DataFrame vide ou None pour {pair_key} {tf}"
                                )

                    print("[DEBUG] Structure finale de bot.indicators :")
                    for k, v in bot.indicators.items():
                        print(f"  {k}: {list(v.keys()) if isinstance(v, dict) else v}")

                    bot.save_shared_data()

                    # Calcul de la durée et rapports
                    duration = (datetime.utcnow() - start).total_seconds()
                    print(f"✅ Cycle terminé en {duration:.1f}s")

                    # Envoi des mises à jour
                    await send_cycle_reports(
                        bot, trade_decisions, cycle, regime, duration
                    )

                except Exception as e:
                    error_msg = f"⚠️ Erreur cycle {cycle}: {e}"
                    print(f"[DEBUG] Exception in main trading loop: {e}")
                    logger.error(error_msg)
                    await bot.telegram.send_message(error_msg)

                # Attente avant le prochain cycle
                await asyncio.sleep(30)

        except KeyboardInterrupt:
            await handle_shutdown(bot, "👋 Bot arrêté proprement")
        except Exception as e:
            await handle_shutdown(bot, f"💥 Erreur fatale: {e}")

    # Fonctions auxiliaires pour le traitement des données et l'analyse


def prepare_ohlcv_data(ohlcv_data):
    """Prépare les données OHLCV pour l'analyse"""
    try:
        if not all(
            k in ohlcv_data
            for k in ["open", "high", "low", "close", "volume", "timestamp"]
        ):
            return None

        return pd.DataFrame(
            {
                0: ohlcv_data["timestamp"],
                1: ohlcv_data["open"],
                2: ohlcv_data["high"],
                3: ohlcv_data["low"],
                4: ohlcv_data["close"],
                5: ohlcv_data["volume"],
            }
        )
    except Exception as e:
        logging.error(f"Erreur préparation OHLCV: {e}")
        return None


async def calculate_combined_score(bot, data, signal, pair):
    """Calcule le score combiné des différents signaux"""
    try:
        combined_score = 0

        # Signal technique
        if signal["action"] == "buy":
            combined_score += signal["confidence"] * 0.5
        elif signal["action"] == "sell":
            combined_score -= signal["confidence"] * 0.5

        # Signal IA
        if bot.ai_enabled:
            ai_signal = data.get("ai_prediction", 0.5)
            combined_score += (ai_signal - 0.5) * 2 * bot.ai_weight

        # Analyse des news
        if bot.news_enabled:
            sentiment_score = data.get("sentiment", 0)
            if sentiment_score != 0:
                sentiment_weight = calculate_sentiment_weight(
                    bot, data, sentiment_score
                )
                combined_score += sentiment_score * sentiment_weight

        return combined_score

    except Exception as e:
        logging.error(f"Erreur calcul score: {e}")
        return 0


def calculate_sentiment_weight(bot, data, sentiment_score):
    """Calcule le poids du sentiment en fonction de son intensité et de son âge"""
    try:
        # Poids de base
        sentiment_weight = bot.news_weight * (1 + abs(sentiment_score))

        # Amplification pour sentiments forts
        if abs(sentiment_score) > 0.7:
            sentiment_weight *= 1.5

        # Facteur temporel
        time_factor = 1.0
        if "sentiment_timestamp" in data:
            elapsed_time = time.time() - data["sentiment_timestamp"]
            time_factor = max(0.2, 1.0 - (elapsed_time / (3600 * 12)))

        return sentiment_weight * time_factor

    except Exception as e:
        logging.error(f"Erreur calcul poids sentiment: {e}")
        return bot.news_weight


async def generate_trade_decision(bot, pair, combined_score, data, signal):
    """Génère une décision de trading basée sur le score combiné"""
    try:
        # Détermination de l'action
        final_action = "neutral"
        if combined_score > 0.3:
            final_action = "buy"
        elif combined_score < -0.3:
            final_action = "sell"

        # Calcul de la confiance
        confidence = min(0.99, abs(combined_score) + 0.5)

        # Logging de la décision
        print(f"📡 {pair}: {final_action.upper()} ({confidence:.0%})")

        # Préparation de la décision
        return {
            "pair": pair,
            "action": final_action,
            "confidence": confidence,
            "signals": {
                "technical": signal["confidence"],
                "ai": data.get("ai_prediction", 0.5),
                "sentiment": data.get("sentiment", 0),
            },
        }

    except Exception as e:
        logging.error(f"Erreur génération décision: {e}")
        return None


async def handle_arbitrage_opportunities(bot):
    """Gère la détection et l'exécution des opportunités d'arbitrage"""
    try:
        opportunities = await bot.detect_arbitrage_opportunities()
        if not opportunities:
            return

        print(f"💹 {len(opportunities)} opportunités d'arbitrage détectées")

        for opp in opportunities:
            # Logging de l'opportunité
            print(
                f"  • {opp['pair']}: {opp['diff_percent']:.2f}% entre "
                f"{opp['exchange1']} et {opp['exchange2']}"
            )

            # Notification Telegram
            await bot.telegram.send_arbitrage_alert(opp)

            # Exécution si profitable
            if opp["diff_percent"] > 0.5:
                print(f"🔄 Exécution de l'arbitrage sur {opp['pair']}")
                await bot.execute_arbitrage(opp)

    except Exception as e:
        logging.error(f"Erreur gestion arbitrage: {e}")


async def execute_trade_decisions(bot, trade_decisions):
    """Exécute les décisions de trading"""
    try:
        for decision in trade_decisions:
            if not bot.is_live_trading or abs(decision["confidence"]) <= 0.5:
                continue

            pair = decision["pair"]
            pair_key = pair.replace("/", "").upper()
            side = "BUY" if decision["action"] == "buy" else "SELL"

            # Calcul du montant
            amount = calculate_position_size(bot, decision)

            # Vérification des stop-loss
            if bot.check_stop_loss(pair_key, side):
                print(f"⚠️ Stop-loss actif pour {pair}, ordre annulé")
                continue

            # Exécution de l'ordre
            trade_result = await bot.execute_trade(pair_key, side, amount)

            # Notification du résultat
            if trade_result["status"] == "completed":
                await send_trade_notification(bot, decision, trade_result, amount)

    except Exception as e:
        logging.error(f"Erreur exécution trades: {e}")


async def run_automl_tuning(bot, mode="cnn_lstm"):
    """Lance une optimisation AutoML/Optuna complète (manuelle ou auto)"""
    print("🔬 Lancement AutoML/Optuna...")
    import time

    start = time.time()
    if mode == "cnn_lstm":
        best_params = tune_hyperparameters()
        print("✅ Optuna tuning terminé. Meilleurs hyperparams:", best_params)
    elif mode == "full":
        best_trials = optimize_hyperparameters_full()
        print("✅ Optuna full tuning terminé. Résumé:", best_trials)
    else:
        print("❌ Mode AutoML inconnu")
        return
    duration = time.time() - start
    print(f"Durée optimisation: {duration:.1f}s")
    # (Optionnel) Recharge config/model avec les meilleurs params
    # bot.reload_model(best_params) ou autre logique
    return best_params if mode == "cnn_lstm" else best_trials


def calculate_position_size(bot, decision):
    """Calcule la taille de position optimale"""
    try:
        base_amount = 0.01
        volatility_factor = decision.get("signals", {}).get("volatility", 0.5)

        # Ajustement par la volatilité
        risk_adjusted = base_amount * (1 - volatility_factor * 0.5)

        # Ajustement par la confiance
        signal_adjusted = risk_adjusted * (0.5 + decision["confidence"] * 0.5)

        return signal_adjusted

    except Exception as e:
        logging.error(f"Erreur calcul taille position: {e}")
        return 0.01


async def send_trade_notification(bot, decision, trade_result, amount):
    """Envoie une notification pour un trade exécuté"""
    try:
        message = (
            f"🔄 <b>Trade exécuté</b>\n\n"
            f"📊 Paire: {decision['pair']}\n"
            f"📈 Action: {decision['action'].upper()}\n"
            f"💰 Montant: {amount}\n"
            f"🎯 Confiance: {decision['confidence']:.0%}\n"
            f"💵 Prix: {trade_result.get('avg_price', 'N/A')}\n\n"
            f"🧠 Signaux:\n"
            f"  • Technique: {decision['signals']['technical']:.0%}\n"
            f"  • IA: {decision['signals']['ai']:.2f}\n"
            f"  • Sentiment: {decision['signals']['sentiment']:.2f}"
        )
        await bot.telegram.send_message(message)

    except Exception as e:
        logging.error(f"Erreur envoi notification: {e}")


async def send_cycle_reports(bot, trade_decisions, cycle, regime, duration):
    """Envoie les rapports de fin de cycle"""
    try:
        # Mise à jour Telegram standard
        await bot.send_telegram_updates()

        # Rapport des trades si nécessaire
        if trade_decisions:
            trade_report = "💹 <b>Trades exécutés</b>\n\n"
            for trade in trade_decisions:
                status = trade.get("result", {}).get("status", "pending")
                emoji = (
                    "✅"
                    if status == "completed"
                    else "⚠️" if status == "simulated" else "❌"
                )
                trade_report += f"{emoji} {trade['pair']}: {trade['action'].upper()} ({trade['confidence']:.0%})\n"

            await bot.telegram.send_message(trade_report)
        else:
            await bot.telegram.send_cycle_update(cycle, regime, duration)

    except Exception as e:
        logging.error(f"Erreur envoi rapports: {e}")


async def handle_shutdown(bot, message):
    """Gère l'arrêt propre du bot"""
    try:
        print(f"\n{message}")
        await bot.telegram.send_message(message)
        await bot.ws_collector.stop()
        bot.save_shared_data()
    except Exception as e:
        logging.error(f"Erreur arrêt bot: {e}")


if __name__ == "__main__":
    import sys
    from src.backtesting.core.backtest_engine import BacktestEngine
    from src.strategies import sma_strategy, breakout_strategy, arbitrage_strategy

    # --- 1. Argument parsing avancé
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backtest", action="store_true", help="Lancer un backtest quantitatif"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/historical/BTCUSDT_1h.csv",
        help="Chemin du CSV market data",
    )
    parser.add_argument("--capital", type=float, default=10000, help="Capital initial")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sma",
        choices=["sma", "breakout", "arbitrage"],
        help="Stratégie à utiliser",
    )
    # Ajoute ici d'autres paramètres si besoin...
    args, unknown = parser.parse_known_args()

    # --- 2. Mode AutoML/Tuning (prioritaire sur tout le reste)
    if "automl" in sys.argv or "tune" in sys.argv:
        import asyncio

        asyncio.run(run_automl_tuning(None, mode="cnn_lstm"))

    # --- 3. Mode backtest CLI
    elif args.backtest:
        print("=== Lancement du backtesting quantitatif ===")
        # 1. Charge les paires depuis la config
        config_path = "config/trading_pairs.json"
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            pairs = config.get("valid_pairs", ["BTC/USDT"])
        except Exception as e:
            print("Impossible de charger la config, on utilise BTC/USDT.")
            pairs = ["BTC/USDT"]

        # 2. Définis la période à backtester
        nb_days = 30
        end_dt = pd.Timestamp.utcnow()
        start_dt = end_dt - pd.Timedelta(days=nb_days)
        interval = Client.KLINE_INTERVAL_1HOUR

        # 3. Stratégies
        strategy_map = {
            "sma": sma_strategy,
            "breakout": breakout_strategy,
            "arbitrage": arbitrage_strategy,
        }
        strategy_func = strategy_map.get(args.strategy, sma_strategy)

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        for pair in pairs:
            symbol = pair.replace("/", "")
            print(f"Téléchargement des données pour {symbol}...")
            df = fetch_binance_ohlcv(
                symbol,
                interval,
                start_dt.strftime("%d %b %Y"),
                end_dt.strftime("%d %b %Y"),
                api_key=api_key,
                api_secret=api_secret,
            )
            if df is None or len(df) == 0:
                print(f"Données manquantes pour {pair}, backtest ignoré.")
                continue

            engine = BacktestEngine(initial_capital=args.capital)
            print(f"Backtest sur {pair} ({len(df)} lignes)...")
            results = engine.run_backtest(df, strategy_func)
            print(f"Résultats du backtest pour {pair} :")
            print(results)
        sys.exit(0)
    else:
        import asyncio
        asyncio.run(run_clean_bot())
