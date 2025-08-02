import warnings

# Supprimer TOUS les warnings Python
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
import json
import asyncio
import aiohttp
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import argparse
import pandas as pd
import pandas_ta as pta
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import json
import lz4.frame
import shutil

from decimal import Decimal
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.analysis.news.cointelegraph_fetcher import fetch_cointelegraph_news
from src.analysis.news.sentiment_analyzer import NewsSentimentAnalyzer

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

from src.ai.auto_strategy_generator import auto_generate_and_backtest
from src.ai.auto_strategy_generator import appliquer_config_strategy
from src.ai.train_cnn_lstm import train_with_live_data
from src.ai.deep_learning_model import features_to_array

from collections import defaultdict

from deep_translator import GoogleTranslator
from src.ai.hybrid_model import HybridAI

from bingx_order_executor import BingXOrderExecutor
from src.exchanges.bingx_exchange import BingXExchange

from src.risk_tools import kelly_criterion, calculate_var, calculate_max_drawdown

from src.portfolio.position_sizer import dynamic_position_size, compute_drawdown

from src.portfolio.exit_manager import ExitManager

from src.analysis.filters.volatility_anomaly_filter import filter_market

from src.analysis.filters.correlation_filter import filter_uncorrelated_pairs

from src.risk_tools.news_pause_manager import NewsPauseManager

from src.portfolio.binance_utils import get_avg_entry_price_binance_spot

# Charger les variables d'environnement depuis .env
load_dotenv()

LOG_FILE = "src/bot_logs.txt"


class ExchangeConnector:
    """
    Abstraction pour gérer plusieurs exchanges facilement.
    Chaque exchange doit avoir un client Python (Binance, Kucoin, OKX...).
    Tu utilises cette classe pour faire les ordres et récupérer le portefeuille.
    """

    def __init__(self, name, client=None):
        self.name = name
        self.client = client

    def execute_order(self, symbol, side, amount, **kwargs):
        if self.name == "binance":
            # Exemple simplifié, adapte à ton SDK
            return self.client.create_order(
                symbol=symbol, side=side, quantity=amount, **kwargs
            )
        elif self.name == "kucoin":
            # Placeholder, à implémenter
            pass
        elif self.name == "okx":
            # Placeholder, à implémenter
            pass

    def get_portfolio(self):
        if self.name == "binance":
            return self.client.get_account()
        elif self.name == "kucoin":
            pass
        elif self.name == "okx":
            pass

    def get_orderbook(self, symbol):
        if self.name == "binance":
            return self.client.get_order_book(symbol=symbol)
        elif self.name == "kucoin":
            pass
        elif self.name == "okx":
            pass


def add_dl_features(df):
    """
    Ajoute les features 'rsi', 'macd', 'volatility' nécessaires à l'entraînement IA.
    Corrige intelligemment les NaN/inf au lieu de tout drop ou reset.
    """

    # Tri par timestamp pour éviter des NaN liés au mauvais ordre
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset="timestamp", keep="last")

    # RSI 14
    if "rsi" not in df or df["rsi"].isnull().all():
        try:
            if len(df) >= 15:
                df["rsi"] = pta.rsi(df["close"], length=14)
            else:
                df["rsi"] = np.nan
        except Exception:
            df["rsi"] = np.nan
    # MACD
    if "macd" not in df or df["macd"].isnull().all():
        try:
            if len(df) >= 27:
                macd = pta.macd(df["close"])
                df["macd"] = macd["MACD_12_26_9"] if "MACD_12_26_9" in macd else np.nan
            else:
                df["macd"] = np.nan
        except Exception:
            df["macd"] = np.nan
    # Volatility
    if "volatility" not in df or df["volatility"].isnull().all():
        try:
            if len(df) >= 15:
                returns = np.log(df["close"]).diff()
                df["volatility"] = returns.rolling(14).std()
            else:
                df["volatility"] = np.nan
        except Exception:
            df["volatility"] = np.nan

    # Nettoyage intelligent NaN/inf (ffill puis bfill puis 0)
    for col in ["rsi", "macd", "volatility"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
    return df


def log_dashboard(message):
    print(message)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} | {message}\n")
    except Exception as e:
        print(f"[LOG ERROR] {e}")


def _generate_analysis_report(
    indicators_analysis, regime, news_sentiment=None, trade_decisions=None
):

    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        "📊 Analyse complète du marché:",
        f"Date: {current_time} UTC",
        f"Régime: {regime}",
        "\nTendances principales:",
    ]
    # Analyse des news
    if news_sentiment:
        try:
            sentiment = float(news_sentiment.get("overall_sentiment", 0) or 0)
            impact = float(news_sentiment.get("impact_score", 0) or 0)
            major_events = news_sentiment.get("major_events", "Aucun")
            report.extend(
                [
                    "\n📰 Analyse des News:",
                    f"Sentiment: {sentiment:.2%}",
                    f"Impact estimé: {impact:.2%}",
                    f"Événements majeurs: {major_events}",
                ]
            )
        except Exception as e:
            report.append(f"\n📰 Erreur sur analyse news : {e}")
    else:
        report.append("\n📰 Analyse des News: Aucune donnée disponible.")

    major_news = news_sentiment.get("latest_news", []) if news_sentiment else []
    if major_news:
        report.append("Dernières news :")
        for news in major_news[:3]:
            report.append(f"- {news}")

    for timeframe, analysis in indicators_analysis.items():
        try:
            report.append(f"\n⏰ {timeframe}:")
            trend_strength = float(
                analysis.get("trend", {}).get("trend_strength", 0) or 0
            )
            volatility = float(
                analysis.get("volatility", {}).get("current_volatility", 0) or 0
            )
            volume_profile = analysis.get("volume", {}).get("volume_profile", {})
            # Cohérence volume (float ou dict)
            if isinstance(volume_profile, dict):
                volume_strength = volume_profile.get("strength", "N/A")
            else:
                volume_strength = volume_profile
            report.extend(
                [
                    f"- Force de la tendance: {trend_strength:.2%}",
                    f"- Volatilité: {volatility:.2%}",
                    f"- Volume: {volume_strength}",
                    f"- Signal dominant: {analysis.get('dominant_signal', 'Neutre')}",
                ]
            )
            if trade_decisions and timeframe in trade_decisions:
                dec = trade_decisions[timeframe]
                try:
                    confidence = float(dec.get("confidence", 0))
                    tech = float(dec.get("tech", 0))
                    ia = float(dec.get("ai", 0))
                    sentiment_trade = float(dec.get("sentiment", 0))
                except Exception:
                    confidence = tech = ia = sentiment_trade = 0.0
                report.append(
                    f"└─ 🎯 Décision de trade: {dec['action'].upper()} "
                    f"(Conf: {confidence:.2f}, "
                    f"Tech: {tech:.2f}, "
                    f"IA: {ia:.2f}, "
                    f"Sentiment: {sentiment_trade:.2f})"
                )
        except Exception as e:
            report.extend(
                [
                    f"\n⏰ {timeframe}:",
                    "- Données non disponibles",
                    "- Analyse en cours...",
                ]
            )
    return "\n".join(report)


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
    # 🔷 Tri systématique par timestamp après chargement OHLCV
    df = df.sort_values("timestamp").reset_index(drop=True)
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
    parser.add_argument("--capital", type=float, default=0, help="Capital initial")
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
        log_dashboard("=== Lancement du backtesting quantitatif ===")
        df = pd.read_csv(args.data)
        # 🔷 Tri par timestamp pour cohérence des indicateurs
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

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
        log_dashboard("Résultats backtest :")
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
    utc_now = datetime.utcnow()
    polynesie_offset = timedelta(hours=-10)
    local_dt = utc_now + polynesie_offset
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")


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
        self.N_FEATURES = 8
        self.N_STEPS = 63

    def get_input_dim(self):
        return self.N_FEATURES * self.N_STEPS * len(self.pairs_valid)

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

        MAX_TELEGRAM_LENGTH = 4000
        if len(full_message) > MAX_TELEGRAM_LENGTH:
            full_message = (
                full_message[: MAX_TELEGRAM_LENGTH - 20]
                + "\n... (troncature automatique)"
            )

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
        """Envoie un message unique et lisible pour chaque trade exécuté"""
        emoji = (
            "🟢"
            if trade_data.get("side", "").upper() == "BUY"
            else "🔴" if trade_data.get("side", "").upper() == "SELL" else "⚪️"
        )
        message = (
            f"{emoji} <b>TRADE EXÉCUTÉ</b>\n\n"
            f"📊 Paire : {trade_data.get('symbol','?')}\n"
            f"Action : <b>{trade_data.get('side','?')}</b>\n"
            f"Montant : {trade_data.get('amount','?')}\n"
            f"Prix : {trade_data.get('price','?')}\n"
            f"Total : {trade_data.get('total', 'N/A')}\n"
            f"Confiance : {trade_data.get('confidence', 'N/A')}\n"
            f"Signaux : Tech {trade_data.get('tech', 'N/A')} | IA {trade_data.get('ia', 'N/A')} | Sentiment {trade_data.get('sentiment', 'N/A')}\n"
            f"Raison : {trade_data.get('reason', 'Signal de trading')}\n"
        )
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

        def real_translate_title(title):
            try:
                return GoogleTranslator(source="auto", target="fr").translate(title)
            except Exception:
                return title

        def translate_title(title):
            original = title
            dico = {
                "Bitcoin": "Bitcoin",
                "Ethereum": "Ethereum",
                "price": "prix",
                "update": "mise à jour",
                "reaches": "atteint",
                "falls": "chute",
                "surges": "explose",
                "network": "réseau",
                "record": "record",
                "launch": "lancement",
                "approval": "approbation",
                "hack": "piratage",
                "coin": "jeton",
                "exchange": "plateforme",
                "regulation": "réglementation",
                "ETF": "ETF",
                "market": "marché",
                "crash": "effondrement",
                "rise": "hausse",
                "buy": "achat",
                "sell": "vente",
                "token": "jeton",
                "trading": "trading",
                "volume": "volume",
                "support": "support",
                "resistance": "résistance",
            }
            for en, fr in dico.items():
                title = title.replace(en, fr)

            if title == original:
                try:
                    from deep_translator import GoogleTranslator

                    return GoogleTranslator(source="auto", target="fr").translate(title)
                except Exception:
                    return title

            return title

        # Remplacer [:5] par rien pour prendre tous les titres
        for news in filtered_news:
            src = news.get("source", "default")
            emoji = source_emoji.get(src, source_emoji["default"])
            title = news.get("title", "NO_TITLE")
            url = news.get("url", "")
            # Traduction simplifiée
            fr_title = real_translate_title(title)
            if url:
                title_line = f'{emoji} <a href="{url}">{fr_title}</a>'
            else:
                title_line = f"{emoji} {fr_title}"
            title_line += f" <i>({src})</i>\n"
            message += title_line

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


def get_sentiment_summary_from_batch(sentiment_scores, top_n=5):
    import numpy as np

    # Filtre les news avec score
    valid = [
        item
        for item in sentiment_scores
        if "sentiment" in item and item["sentiment"] is not None
    ]
    if not valid:
        return {
            "sentiment_global": 0.0,
            "n_news": 0,
            "top_symbols": [],
            "top_news": [],
        }
    # Calcul de la moyenne pondérée
    sentiments = [item["sentiment"] for item in valid]
    sentiment_global = float(np.mean(sentiments))
    # Top news (par score absolu)
    top_news = sorted(valid, key=lambda x: abs(x["sentiment"]), reverse=True)[:top_n]
    top_news_titles = [news["title"] for news in top_news if "title" in news]
    # Top symbols (fréquence + score fort)
    symbol_scores = {}
    for item in valid:
        for s in item.get("symbols", []):
            symbol_scores.setdefault(s, []).append(item["sentiment"])
    top_symbols = sorted(
        symbol_scores.items(), key=lambda kv: abs(np.mean(kv[1])), reverse=True
    )
    top_symbols = [s for s, scores in top_symbols[:top_n]]
    return {
        "sentiment_global": sentiment_global,
        "n_news": len(valid),
        "top_symbols": top_symbols,
        "top_news": top_news_titles,
    }


def merge_news_processed(old_scores, new_scores):
    """Merge les news existantes (ayant 'processed') avec les nouvelles, en préservant ce flag par titre."""
    old_map = {
        n.get("title"): n.get("processed", False) for n in old_scores if "title" in n
    }
    for n in new_scores:
        if n.get("title") in old_map:
            n["processed"] = old_map[n.get("title")]
    return new_scores


class APIRequestOptimizer:
    """Gestionnaire optimisé des requêtes API"""

    def __init__(self):
        self.rate_limits = {}
        self.cache = TTLCache(maxsize=100, ttl=60)
        self.backup_endpoints = []

    async def execute_with_retry(self, request_func, max_retries=3):
        for i in range(max_retries):
            try:
                return await request_func()
            except Exception as e:
                if i == max_retries - 1:
                    raise e
                await asyncio.sleep(2**i)


class DataBackupManager:
    """Gestionnaire de sauvegarde des données"""

    def __init__(self, backup_dir="backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def backup_trade_data(self, data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_data_{timestamp}.parquet"
        self.save_parquet(data, os.path.join(self.backup_dir, filename))


class PerformanceMonitor:
    """Système avancé de monitoring des performances"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []

    def track_metric(self, name, value):
        self.metrics[name].append({"timestamp": datetime.now(), "value": value})
        self.check_alerts(name, value)

    def check_alerts(self, metric_name, value):
        """Vérifie les déviations de performance"""
        if metric_name == "win_rate" and value < 0.55:
            self.add_alert("Win rate below threshold", severity="high")
        elif metric_name == "drawdown" and value < -0.15:
            self.add_alert("Excessive drawdown", severity="critical")


class TradingBotM4:
    def __init__(self):
        # Configuration de base existante...
        self.config = {
            "TRADING": {
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "pairs": [
                    "BTC/USDC",
                    "ETH/USDC",
                    "LTC/USDC",
                    "XRP/USDC",
                    "DOGE/USDC",
                    "BNB/USDC",
                    "ADA/USDC",
                    "SOL/USDC",
                    "TRX/USDC",
                    "SUI/USDC",
                ],
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
            "news": {
                "sentiment_weight": 0.15,
                "update_interval": 300,
                "storage_path": "data/news_analysis.json",
                "low_watermark_ratio": 0.2,
                "symbol_mapping": {
                    "bitcoin": "BTC",
                    "ethereum": "ETH",
                    "cardano": "ADA",
                    "solana": "SOL",
                    "litecoin": "LTC",
                    "xrp": "XRP",
                    "dogecoin": "DOGE",
                    "binancecoin": "BNB",
                    "tron": "TRX",
                    "sui": "SUI",
                    "stablecoin": "USDT",
                    "ink": "INK",
                    "ena": "ENA",
                    "ledger": "BTC",
                    "tether": "USDT",
                },
            },
        }

        self.last_correlation_check = 0
        self.correlation_cache = {}
        self.correlation_cache_ttl = 300  # 5 minutes

        self.system_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "api_latency": [],
            "ws_status": True,
        }

        self.news_pause_manager = NewsPauseManager(
            default_pause_cycles=6
        )  # 6 cycles = 3 minutes si cycle=30s

        self.exit_manager = ExitManager(
            tp_levels=[(0.03, 0.3), (0.07, 0.3)], trailing_pct=0.03
        )

        self.trade_decisions = {}

        self.signal_fusion_params = self.load_signal_fusion_params()

        self.positions = {}  # Ajouté : gestion des positions spot par paire
        self.stop_loss_pct = 0.03  # 3% stop-loss, modifiable

        bingx_api_key = os.getenv("BINGX_API_KEY")
        bingx_api_secret = os.getenv("BINGX_API_SECRET")

        self.bingx_client = BingXExchange(
            bingx_api_key, bingx_api_secret
        )  # adapte selon ton code
        self.bingx_executor = BingXOrderExecutor(self.bingx_client)

        # --- SYNCHRONISATION AUTO DES PAIRS ---
        self.pairs_valid = self.config["TRADING"]["pairs"]

        # --- WS COLLECTOR --- (toujours synchro avec la config)
        self.ws_collector = BufferedWSCollector(
            symbols=[s.replace("/", "").upper() for s in self.pairs_valid],
            timeframes=self.config["TRADING"]["timeframes"],
            maxlen=2000,
        )
        # Initialize basic attributes...
        self.data_file = SHARED_DATA_PATH
        self.current_cycle = 0
        self.regime = MARKET_REGIMES["RANGING"]
        self.market_data = {}
        self.indicators = {}
        self.news_analyzer = NewsSentimentAnalyzer(self.config)
        self.news_enabled = True
        self.dl_model_last_mtime = None

        self.news_weight = 0.15
        self.ai_weight = 0.5
        self.ensure_float = lambda x: (
            float(x) if isinstance(x, (int, float, str)) else 0.0
        )
        self.technical_weight = 0.6  # Poids des signaux techniques (60%)
        self.ai_enabled = False
        self.pairs_valid = self.config["TRADING"]["pairs"]

        # Initialisation de l'arbitrage engine
        try:
            self.arbitrage_engine = ArbitrageEngine()
            self.brokers = self.arbitrage_engine.brokers
            log_dashboard("✅ ArbitrageEngine initialisé avec succès")
        except Exception as e:
            log_dashboard(f"⚠️ Erreur initialisation ArbitrageEngine: {e}")
            self.arbitrage_engine = None
            self.brokers = {}

        self.arbitrage_executor = ArbitrageExecutor(self.brokers)

        # Initialisation de l'environnement (une seule fois)
        print("Configuration de l'environnement...")

        # --- ENVIRONNEMENT TRADING ---
        self.env = TradingEnv(
            trading_pairs=self.pairs_valid,
            timeframes=self.config["TRADING"]["timeframes"],
        )
        print("✅ Environnement initialisé avec succès")

        # Initialisation de l'IA (modèle réel uniquement)
        self._initialize_ai()

        # Initialise les données partagées
        self.initialize_shared_data()

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
            "min_profit": 0.5,
            "max_exposure": 1000,
            "enabled_exchanges": ["binance", "kucoin", "huobi"],
        }
        # Sécurité avancée: gestion de clé cold wallet
        # Ajoute cette option (True = utilisation automatique, False = ignorée)
        self.use_cold_wallet_key = False  # ou False selon besoin

        self.key_manager = KeyManager()
        if self.use_cold_wallet_key:
            if not self.key_manager.has_key():
                print(
                    "Aucune clé cold wallet détectée, génération d'une nouvelle clé sécurisée…"
                )
                pk = self.key_manager.generate_private_key()
                self.key_manager.save_private_key()
                print("Clé cold wallet générée et sauvegardée de manière chiffrée.")
            else:
                try:
                    # Si tu veux demander le mot de passe à chaque fois (optionnel):
                    # password = self.ask_wallet_password()
                    # self.key_manager.load_private_key(password=password)
                    self.key_manager.load_private_key()
                    print("Clé cold wallet chargée avec succès.")
                except Exception as e:
                    print(f"Erreur de chargement de la clé cold wallet: {e}")
        else:
            print("⚠️ Utilisation de la clé cold wallet désactivée.")

        self.auto_strategy_config = None
        if os.path.exists("config/auto_strategy.json"):
            with open("config/auto_strategy.json", "r") as f:
                self.auto_strategy_config = json.load(f)
            log_dashboard("✅ Auto-stratégie chargée :", self.auto_strategy_config)
        self.sync_positions_with_binance()

    def calculate_correlation_matrix(self):
        """Calcule la matrice de corrélation entre les paires"""
        correlations = {}
        for pair1 in self.pairs_valid:
            for pair2 in self.pairs_valid:
                correlation = self.calculate_pair_correlation(pair1, pair2)
                correlations[f"{pair1}-{pair2}"] = correlation
        return correlations

    def adjust_position_sizing(self, base_size, correlation_factor):
        """Ajuste le sizing selon les corrélations"""
        return base_size * (1 - correlation_factor)

    def weighted_signal_fusion(self, signals):
        """Fusion pondérée des signaux avec poids adaptatifs"""
        weights = {"technical": 0.4, "ai": 0.3, "sentiment": 0.2, "orderflow": 0.1}
        total_score = 0
        for signal_type, value in signals.items():
            if signal_type in weights:
                total_score += value * weights[signal_type]
        return total_score

    def track_advanced_metrics(self):
        """Suivi de métriques avancées"""
        metrics = {
            "sharpe_ratio": self.calculate_sharpe(),
            "sortino_ratio": self.calculate_sortino(),
            "calmar_ratio": self.calculate_calmar(),
            "win_rate": self.get_win_rate(),
            "avg_profit": self.get_avg_profit(),
            "max_drawdown": self.get_max_drawdown(),
        }
        return metrics

    def safe_trade_execution(self, order):
        """Exécution sécurisée des ordres"""
        try:
            # Vérifications pré-trade
            self.check_margin_requirements()
            self.verify_risk_limits()
            self.check_market_conditions()

            # Exécution avec retry
            for attempt in range(3):
                try:
                    result = self.execute_order(order)
                    return result
                except ConnectionError:
                    continue

        except Exception as e:
            self.logger.error(f"Erreur exécution: {e}")
            return None

    # Backup automatique
    def backup_critical_data(self):
        try:
            backup_data = {
                "timestamp": get_current_time(),
                "positions": self.positions_binance,
                "market_data": self.market_data,
                "indicators": self.indicators,
                "system_metrics": self.system_metrics,
                "performance": self.get_performance_metrics(),
            }

            # Sauvegarde compressée
            import lz4.frame
            import json

            backup_path = (
                f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.lz4"
            )
            with lz4.frame.open(backup_path, "wb") as f:
                f.write(json.dumps(backup_data).encode())

            # Nettoyage des vieux backups (garde 7 jours)
            self.cleanup_old_backups(days=7)

            return True
        except Exception as e:
            self.logger.error(f"Erreur backup: {e}")
            return False

    def monitor_system_health(self):
        try:
            import psutil

            metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "api_latency": self.measure_api_latency(),
                "ws_status": self.check_ws_status(),
            }

            # Stockage historique
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.system_metrics[key].append(
                        {"timestamp": get_current_time(), "value": value}
                    )

            return metrics
        except Exception as e:
            self.logger.error(f"Erreur monitoring système: {e}")
            return {}

    def analyze_volume_profile(self, symbol, timeframe="1h"):
        """Analyse avancée du volume profile"""
        df = self.get_timeframe_data(symbol, timeframe)
        if df is None:
            return None

        # Calcul des points d'accumulation/distribution
        volume_nodes = self.calculate_volume_nodes(df)

        # Détection des zones de haute liquidité
        liquidity_zones = self.identify_liquidity_zones(df)

        return {
            "volume_nodes": volume_nodes,
            "liquidity_zones": liquidity_zones,
            "poc_price": self.calculate_poc_price(df),
        }

    def analyze_order_pressure(self, symbol):
        """Analyse de la pression des ordres limites"""
        orderbook = self.get_ws_orderbook(symbol)
        if not orderbook:
            return None

        bid_pressure = self.calculate_bid_pressure(orderbook["bids"])
        ask_pressure = self.calculate_ask_pressure(orderbook["asks"])

        return {
            "bid_pressure": bid_pressure,
            "ask_pressure": ask_pressure,
            "imbalance": bid_pressure - ask_pressure,
        }

    def calculate_dynamic_stoploss(self, symbol, timeframe="1h"):
        """Calcule un stop-loss dynamique basé sur l'ATR"""
        try:
            df = self.get_timeframe_data(symbol, timeframe)
            if df is None:
                return self.stop_loss_pct  # Retourne le stop-loss par défaut

            atr = self.calculate_atr(df, period=14)
            price = df["close"][-1]

            # Stop-loss adaptatif : entre 1% et 3% selon l'ATR
            atr_pct = atr / price
            dynamic_sl = min(max(atr_pct * 2, 0.01), 0.03)

            return dynamic_sl
        except Exception as e:
            self.logger.error(f"Erreur calcul stop-loss dynamique: {e}")
            return self.stop_loss_pct

    def analyze_correlations(self):
        """Analyse les corrélations entre paires pour la diversification"""
        correlations = {}
        for pair1 in self.pairs_valid:
            correlations[pair1] = {}
            for pair2 in self.pairs_valid:
                if pair1 != pair2:
                    corr = self.calculate_pair_correlation(pair1, pair2)
                    correlations[pair1][pair2] = corr
        return correlations

    def calculate_trade_quality_score(self, trade_data):
        """Score de qualité des trades basé sur multiples facteurs"""
        score = 0

        # Timing d'entrée (proximité support/résistance)
        if self.is_near_key_level(trade_data["symbol"], trade_data["price"]):
            score += 2

        # Volume au moment de l'entrée
        if trade_data.get("volume_ratio", 1) > 1.5:
            score += 1

        # Momentum aligné
        if self.check_momentum_alignment(trade_data):
            score += 1

        # Convergence multi-timeframes
        if self.check_timeframe_alignment(trade_data):
            score += 2

        return score

    def calculate_squeeze_momentum(self, df):
        """
        Calcule l'indicateur TTM Squeeze avec momentum.
        Un squeeze se produit quand la volatilité diminue et la pression augmente.
        """
        try:
            # Keltner Channel
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            mean_tp = typical_price.rolling(window=20).mean()
            atr = (
                pd.DataFrame(
                    {
                        "h-l": df["high"] - df["low"],
                        "h-pc": abs(df["high"] - df["close"].shift(1)),
                        "l-pc": abs(df["low"] - df["close"].shift(1)),
                    }
                )
                .max(axis=1)
                .rolling(window=20)
                .mean()
            )

            keltner_up = mean_tp + (atr * 1.5)
            keltner_down = mean_tp - (atr * 1.5)

            # Bollinger Bands
            std = df["close"].rolling(window=20).std()
            bb_up = mean_tp + (std * 2)
            bb_down = mean_tp - (std * 2)

            # Squeeze Detection
            squeeze_on = (bb_up <= keltner_up) & (bb_down >= keltner_down)

            # Momentum
            highest = df["high"].rolling(window=20).max()
            lowest = df["low"].rolling(window=20).min()
            momentum = df["close"] - ((highest + lowest) / 2)

            return {
                "squeeze_on": squeeze_on.iloc[-1],
                "momentum": momentum.iloc[-1],
                "momentum_change": momentum.diff().iloc[-1],
            }
        except Exception as e:
            self.logger.error(f"Erreur calcul squeeze momentum: {e}")
            return {"squeeze_on": False, "momentum": 0, "momentum_change": 0}

    def analyze_volume_distribution(self, df):
        """
        Analyse la distribution des volumes pour identifier les zones d'accumulation/distribution.
        """
        try:
            # Volume Profile
            price_bins = pd.qcut(df["close"], q=10, duplicates="drop")
            vol_profile = df.groupby(price_bins)["volume"].sum()

            # Volume POC (Point of Control)
            poc_price = vol_profile.idxmax().left

            # Volume Delta
            buy_volume = df["volume"][df["close"] > df["open"]].sum()
            sell_volume = df["volume"][df["close"] < df["open"]].sum()
            delta = buy_volume - sell_volume

            # Accumulation/Distribution
            is_accumulation = delta > 0 and df["close"].iloc[-1] > df["close"].mean()
            is_distribution = delta < 0 and df["close"].iloc[-1] < df["close"].mean()

            return {
                "poc_price": poc_price,
                "volume_delta": delta,
                "is_accumulation": is_accumulation,
                "is_distribution": is_distribution,
                "buy_volume_ratio": (
                    buy_volume / (buy_volume + sell_volume)
                    if (buy_volume + sell_volume) > 0
                    else 0.5
                ),
            }
        except Exception as e:
            self.logger.error(f"Erreur analyse volume: {e}")
            return {
                "poc_price": 0,
                "volume_delta": 0,
                "is_accumulation": False,
                "is_distribution": False,
                "buy_volume_ratio": 0.5,
            }

    def analyze_order_flow(self, df):
        """
        Analyse avancée du flux d'ordres pour détecter la pression acheteur/vendeur.
        """
        try:
            # Calcul de l'absorption
            buying_pressure = (
                (df["close"] - df["low"]) / (df["high"] - df["low"])
            ) * df["volume"]
            selling_pressure = (
                (df["high"] - df["close"]) / (df["high"] - df["low"])
            ) * df["volume"]

            # CVD (Cumulative Volume Delta)
            cvd = (buying_pressure - selling_pressure).cumsum()

            # Imbalance Detection
            imbalance = (
                abs(buying_pressure.mean() - selling_pressure.mean())
                / selling_pressure.mean()
            )

            return {
                "buying_pressure": buying_pressure.iloc[-1],
                "selling_pressure": selling_pressure.iloc[-1],
                "cvd": cvd.iloc[-1],
                "imbalance": imbalance,
                "pressure_ratio": (
                    buying_pressure.iloc[-1] / selling_pressure.iloc[-1]
                    if selling_pressure.iloc[-1] != 0
                    else 1
                ),
            }
        except Exception as e:
            self.logger.error(f"Erreur analyse order flow: {e}")
            return {
                "buying_pressure": 0,
                "selling_pressure": 0,
                "cvd": 0,
                "imbalance": 0,
                "pressure_ratio": 1,
            }

    def identify_market_structure(self, df):
        """
        Identifie la structure du marché (tendance, range, etc.).
        """
        try:
            # Swing points
            n = 5  # lookback period
            highs = df["high"].rolling(window=2 * n + 1, center=True).max()
            lows = df["low"].rolling(window=2 * n + 1, center=True).min()

            # Higher Highs & Lower Lows
            higher_highs = highs.diff() > 0
            lower_lows = lows.diff() < 0

            # Trend Detection
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            trend_strength = (ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]

            # Range Analysis
            atr = (
                pd.DataFrame(
                    {
                        "h-l": df["high"] - df["low"],
                        "h-pc": abs(df["high"] - df["close"].shift(1)),
                        "l-pc": abs(df["low"] - df["close"].shift(1)),
                    }
                )
                .max(axis=1)
                .rolling(window=14)
                .mean()
            )

            is_ranging = atr.iloc[-1] < atr.mean()

            return {
                "trend_strength": trend_strength,
                "is_ranging": is_ranging,
                "higher_highs": higher_highs.iloc[-1],
                "lower_lows": lower_lows.iloc[-1],
                "structure": "range" if is_ranging else "trend",
            }
        except Exception as e:
            self.logger.error(f"Erreur identification structure: {e}")
            return {
                "trend_strength": 0,
                "is_ranging": False,
                "higher_highs": False,
                "lower_lows": False,
                "structure": "unknown",
            }

    def multi_timeframe_analysis(self, symbol, timeframes):
        """
        Analyse multi-timeframes pour confirmation.
        """
        try:
            analysis = {}
            for tf in timeframes:
                df = self.get_timeframe_data(symbol, tf)
                if df is None or len(df) < 50:
                    continue

                # Analyses par timeframe
                structure = self.identify_market_structure(df)
                volume = self.analyze_volume_distribution(df)
                momentum = self.calculate_squeeze_momentum(df)

                analysis[tf] = {
                    "structure": structure,
                    "volume": volume,
                    "momentum": momentum,
                    "alignment": self.check_alignment(df),
                }

            return analysis
        except Exception as e:
            self.logger.error(f"Erreur analyse multi-timeframes: {e}")
            return {}

    def calculate_confirmation_score(self, indicators, mtp_analysis):
        """
        Calcule un score de confirmation basé sur tous les indicateurs.
        """
        try:
            score = 0
            weight_sum = 0

            # 1. Structure de marché (40%)
            if indicators.get("market_structure"):
                ms = indicators["market_structure"]
                if ms["structure"] == "trend":
                    score += 0.4 * abs(ms["trend_strength"])
                weight_sum += 0.4

            # 2. Volume Analysis (30%)
            if indicators.get("volume_profile"):
                vp = indicators["volume_profile"]
                if vp["is_accumulation"]:
                    score += 0.3
                elif vp["is_distribution"]:
                    score -= 0.3
                weight_sum += 0.3

            # 3. Momentum & Squeeze (20%)
            if indicators.get("squeeze_momentum"):
                sq = indicators["squeeze_momentum"]
                if sq["squeeze_on"] and sq["momentum"] > 0:
                    score += 0.2
                elif sq["squeeze_on"] and sq["momentum"] < 0:
                    score -= 0.2
                weight_sum += 0.2

            # 4. Order Flow (10%)
            if indicators.get("order_flow"):
                of = indicators["order_flow"]
                if of["pressure_ratio"] > 1:
                    score += 0.1 * min(of["pressure_ratio"] - 1, 1)
                elif of["pressure_ratio"] < 1:
                    score -= 0.1 * min(1 - of["pressure_ratio"], 1)
                weight_sum += 0.1

            # Normalisation
            if weight_sum > 0:
                score = score / weight_sum

            return score
        except Exception as e:
            self.logger.error(f"Erreur calcul score confirmation: {e}")
            return 0

    def get_volatility_multiplier(self, symbol):
        """
        Retourne un multiplicateur basé sur la volatilité.
        Réduit le sizing quand la volatilité est élevée.
        """
        try:
            df = self.get_recent_data(symbol)
            if df is None or len(df) < 20:
                return 1.0

            # ATR relatif
            atr = (
                pd.DataFrame(
                    {
                        "h-l": df["high"] - df["low"],
                        "h-pc": abs(df["high"] - df["close"].shift(1)),
                        "l-pc": abs(df["low"] - df["close"].shift(1)),
                    }
                )
                .max(axis=1)
                .rolling(window=14)
                .mean()
                .iloc[-1]
            )

            price = df["close"].iloc[-1]
            atr_pct = atr / price

            # Ajustement du multiplicateur
            if atr_pct < 0.01:  # Très faible volatilité
                return 1.2
            elif atr_pct < 0.02:  # Volatilité normale
                return 1.0
            elif atr_pct < 0.03:  # Volatilité élevée
                return 0.8
            elif atr_pct < 0.04:  # Très haute volatilité
                return 0.6
            else:  # Volatilité extrême
                return 0.4

        except Exception as e:
            self.logger.error(f"Erreur calcul multiplicateur volatilité: {e}")
            return 1.0

    def get_risk_multiplier(self, symbol):
        """
        Calcule un multiplicateur de risque basé sur plusieurs facteurs.
        """
        try:
            multiplier = 1.0

            # 1. Corrélation avec le marché
            correlation = self.get_market_correlation(symbol)
            if correlation > 0.8:
                multiplier *= 0.8  # Réduit le risque si forte corrélation

            # 2. Liquidité
            liquidity_score = self.get_liquidity_score(symbol)
            if liquidity_score < 0.5:
                multiplier *= 0.7  # Réduit le risque si faible liquidité

            # 3. Spread moyen
            spread = self.get_average_spread(symbol)
            if spread > 0.001:  # Plus de 0.1%
                multiplier *= 0.9

            # 4. Distance aux supports/résistances
            key_levels = self.get_key_levels(symbol)
            if self.is_near_key_level(symbol, key_levels):
                multiplier *= 1.1  # Augmente légèrement si près d'un niveau clé

            return max(0.3, min(multiplier, 1.2))  # Borne entre 0.3 et 1.2

        except Exception as e:
            self.logger.error(f"Erreur calcul multiplicateur risque: {e}")
            return 1.0

    # Méthodes utilitaires nécessaires
    def get_timeframe_data(self, symbol, timeframe):
        """Helper pour récupérer les données d'un timeframe."""
        try:
            return self.market_data.get(symbol, {}).get(timeframe, None)
        except Exception:
            return None

    def check_alignment(self, df):
        """Vérifie l'alignement des indicateurs."""
        try:
            ema20 = df["close"].ewm(span=20).mean()
            ema50 = df["close"].ewm(span=50).mean()
            rsi = self.calculate_rsi(df["close"])

            price_trend = ema20.iloc[-1] > ema50.iloc[-1]
            momentum_aligned = rsi.iloc[-1] > 50 if price_trend else rsi.iloc[-1] < 50

            return price_trend and momentum_aligned
        except Exception:
            return False

    def safe_update_shared_data(
        self, new_fields: dict, data_file="src/shared_data.json"
    ):
        # 1. Lis le fichier existant SANS jamais repartir sur {}
        try:
            with open(data_file, "r") as f:
                shared_data = json.load(f)
        except Exception:
            # En cas de bug, tente de restaurer une sauvegarde précédente
            backup_file = data_file + ".bak"
            if os.path.exists(backup_file):
                with open(backup_file, "r") as f:
                    shared_data = json.load(f)
            else:
                shared_data = None
        # Si shared_data est None, NE PAS ÉCRIRE !
        if shared_data is None:
            print("[SAFE PATCH] shared_data.json corrompu, skip écriture !")
            return
        # 2. Mets à jour les champs nécessaires
        shared_data.update(new_fields)
        # 3. Sauvegarde une copie de secours avant d’écrire
        try:
            shutil.copyfile(data_file, data_file + ".bak")
        except Exception:
            pass
        # 4. Écris
        with open(data_file, "w") as f:
            json.dump(shared_data, f, indent=4)

    def check_tp_partial(
        self,
        entry_price,
        current_price,
        filled_tp_targets=None,
        tp_levels=[(0.03, 0.3), (0.07, 0.3)],
    ):
        """
        Fractionne la sortie sur plusieurs TP (take profit).
        tp_levels = [(niveau de gain, % à sortir)]
        filled_tp_targets = [bool, bool] selon si les TP ont déjà été touchés
        Retourne (proportion à sortir, new_filled)
        """
        if filled_tp_targets is None:
            filled_tp_targets = [False] * len(tp_levels)
        to_exit = 0
        new_filled = filled_tp_targets[:]
        for i, (tp_pct, frac) in enumerate(tp_levels):
            if (
                not new_filled[i]
                and (current_price - entry_price) / entry_price > tp_pct
            ):
                to_exit += frac
                new_filled[i] = True
        return to_exit, new_filled

    def check_trailing(self, entry_price, price_history, max_price, trailing_pct=0.03):
        """
        Trailing stop universel : sort si le prix retombe de X% par rapport au max atteint.
        """
        if not price_history or len(price_history) < 3:
            return False, max_price
        current_price = price_history[-1]
        if current_price > max_price:
            max_price = current_price
        if current_price < max_price * (1 - trailing_pct):
            return True, max_price
        return False, max_price

    def calculate_atr(df, period=14):
        """Calcul de l'Average True Range (ATR) pour stop-loss dynamique."""
        high = np.array(df["high"])
        low = np.array(df["low"])
        close = np.array(df["close"])
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
        atr = pd.Series(tr).rolling(window=period).mean()
        return float(atr.iloc[-1]) if len(atr) > 0 else 0.01

    def log_closed_position(self, symbol, pos, exit_price, reason):
        closed_position = {
            "symbol": symbol,
            "side": pos.get("side", ""),
            "amount": pos.get("amount", 0),
            "entry_price": pos.get("entry_price"),
            "exit_price": exit_price,
            "pnl_pct": (
                (exit_price - pos.get("entry_price")) / pos.get("entry_price") * 100
                if pos.get("entry_price")
                else 0
            ),
            "pnl_usd": (
                (exit_price - pos.get("entry_price")) * pos.get("amount")
                if pos.get("entry_price")
                else 0
            ),
            "date": datetime.utcnow().isoformat(),
            "reason": reason,
        }
        # Ajoute à closed_positions dans shared_data.json
        closed = []
        try:
            with open(self.data_file, "r") as f:
                shared_data = json.load(f)
                closed = shared_data.get("closed_positions", [])
        except Exception:
            closed = []
        closed.append(closed_position)
        self.safe_update_shared_data({"closed_positions": closed}, self.data_file)

    def get_pending_sales(self):
        """
        Affiche TOUTES les positions spot Binance avec leur état actuel, raison, action du signal, etc.
        Permet d'avoir un état des lieux complet, même si le signal n'est pas SELL.
        """
        pending = []
        now = datetime.utcnow()

        if hasattr(self, "positions_binance"):
            for symbol, pos in self.positions_binance.items():
                entry_price = pos.get("entry_price")
                current_price = pos.get("current_price")
                amount = pos.get("amount")
                pnl_pct = (
                    (current_price - entry_price) / entry_price * 100
                    if entry_price and current_price
                    else 0
                )
                date_achat = None
                temps_en_position = None

                # Récupère le signal du tableau "Scores de décision et signaux"
                td = self.trade_decisions.get(symbol, {})
                action = td.get("action", "neutral")
                confidence = td.get("confidence", None)

                # --- INITIALISATION SYSTÉMATIQUE DES VARIABLES ---
                decision = ""
                pause_status = "Non"
                note = ""

                # Raison
                if action == "SELL":
                    reason = "🔴 Signal SELL"
                    decision = "Vente prévue au prochain cycle"
                elif pnl_pct < -5:
                    reason = (
                        f"🔴 Perte latente {pnl_pct:.1f}%, signal: {action.upper()}"
                    )
                    decision = "Surveillance, risque de vente auto si perte aggrave"
                elif pnl_pct > 7:
                    reason = f"🟢 Gain latent {pnl_pct:.1f}%, signal: {action.upper()}"
                    decision = "Surveillance, possibilité de prise de profit"
                else:
                    reason = f"Signal actuel: {action.upper()}"
                    decision = "Aucune action prévue, position maintenue"

                # Pause (exemple simple, adapte selon tes pauses réelles)
                if hasattr(self, "news_pause_manager"):
                    pauses = self.news_pause_manager.get_active_pauses()
                    if any(symbol in p.get("asset", "") for p in pauses):
                        pause_status = "Oui"
                        note = "Trading suspendu (pause active)"
                elif reason.startswith("🟢 Gain latent"):
                    note = "En zone de profit, TP possible"
                elif reason.startswith("🔴 Perte latente"):
                    note = "Risque de stop-loss"
                else:
                    note = ""

                pending.append(
                    {
                        "symbol": symbol,
                        "reason": reason,
                        "decision": decision,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "amount": amount,
                        "% Gain/Perte": f"{pnl_pct:.2f}%",
                        "temps_en_position_h": (
                            f"{temps_en_position:.1f}"
                            if temps_en_position is not None
                            else "N/A"
                        ),
                        "pause_blocage": pause_status,
                        "note": note,
                    }
                )

        print("DEBUG pending_sales tableau:", pending)
        # Sauvegarde dans shared_data.json
        self.safe_update_shared_data({"pending_sales": pending}, self.data_file)
        return pending

    def get_active_pauses(self):
        """
        Retourne la liste des pauses actives : [{"asset": ..., "action": ..., "cycles_left": ..., "type": ...}, ...]
        """
        pauses = []
        # Recupère les pauses du NewsPauseManager
        for item in self.news_pause_manager.get_active_pauses():
            pauses.append(item)
        return pauses

    def enrich_news_symbols(self, news_list):
        """
        Ajoute automatiquement le champ 'symbols' à chaque news si manquant,
        en utilisant le mapping présent dans la config.
        """
        symbol_mapping = self.config["news"]["symbol_mapping"]
        for news in news_list:
            if "symbols" not in news or not news["symbols"]:
                found_symbols = []
                title = news.get("title", "").lower()
                for k, v in symbol_mapping.items():
                    if k in title:
                        found_symbols.append(v)
                if found_symbols:
                    news["symbols"] = found_symbols
                else:
                    news["symbols"] = []
        return news_list

    def load_signal_fusion_params(self):
        path = "config/best_signal_params.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                params = json.load(f)
            print(f"[OPTIM] Pondérations optimisées chargées : {params}")
            return params
        # Valeurs par défaut
        return {
            "tech_weight": 0.6,
            "ia_weight": 0.3,
            "sentiment_weight": 0.1,
            "buy_threshold": 0.2,
            "sell_threshold": -0.2,
            "mm_risk": 0.05,
        }

    def aggregate_timeframe_signals(self, pair, signals_per_tf):
        """
        Fusionne les signaux multi-timeframes pour une paire donnée.
        signals_per_tf : dict { "1m": {"action":..., "confidence":...}, ... }
        Retourne : action globale ('buy', 'sell', 'neutral') et confiance moyenne pondérée.
        """
        # Pondération : TF + importance (ex : plus fort sur 1h, 4h)
        tf_weights = {"1m": 1, "5m": 2, "15m": 3, "1h": 5, "4h": 4, "1d": 2}
        total_weight = 0
        score = 0
        for tf, d in signals_per_tf.items():
            w = tf_weights.get(tf, 1)
            total_weight += w
            if d["action"] == "buy":
                score += w * d.get("confidence", 0.5)
            elif d["action"] == "sell":
                score -= w * d.get("confidence", 0.5)
            # neutral = 0
        if total_weight == 0:
            return "neutral", 0
        avg_score = score / total_weight
        # Seuils ajustables
        if avg_score > 0.2:
            return "buy", avg_score
        elif avg_score < -0.2:
            return "sell", abs(avg_score)
        else:
            return "neutral", abs(avg_score)

    def sync_positions_with_binance(self):
        if self.is_live_trading and self.binance_client:
            account = self.binance_client.get_account()
            positions = {}
            for bal in account["balances"]:
                asset = bal["asset"]
                free = float(bal["free"])
                if free > 0 and asset not in ("USDC", "USDT"):
                    symbol = f"{asset}/USDC"
                    try:
                        ticker = self.binance_client.get_symbol_ticker(
                            symbol=symbol.replace("/", "")
                        )
                        current_price = float(ticker["price"])
                    except Exception:
                        current_price = None

                    # Utilise uniquement USDC pour entry_price
                    entry_price = get_avg_entry_price_binance_spot(
                        self.binance_client, asset, quote="USDC"
                    )

                    # NE PAS fallback sur current_price !
                    if entry_price is None:
                        entry_price = None
                        pnl_pct = None
                        pnl_usd = None

                    pnl_pct = (
                        (current_price - entry_price) / entry_price * 100
                        if entry_price and current_price
                        else 0.0
                    )
                    pnl_usd = (
                        (current_price - entry_price) * free
                        if entry_price and current_price
                        else 0.0
                    )

                    positions[symbol] = {
                        "side": self.positions.get(symbol, {}).get("side", "long"),
                        "amount": free,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "pnl_pct": pnl_pct,
                        "pnl_usd": pnl_usd,
                        "value_usd": (
                            free * current_price if free and current_price else 0.0
                        ),
                    }
            self.positions_binance = positions

    def is_short(self, symbol):
        return self.positions.get(symbol, {}).get("side") == "short"

    # Ajoute cette méthode pour savoir si on est long
    def is_long(self, symbol):
        # En mode live, ne regarder QUE la position réelle Binance
        if getattr(self, "is_live_trading", False):
            pos_spot = self.positions_binance.get(symbol)
            return pos_spot and float(pos_spot.get("amount", 0)) > 0
        # En simulation, garder la logique actuelle
        return self.positions.get(symbol, {}).get("side") == "long"

    def get_entry_price(self, symbol):
        return self.positions.get(symbol, {}).get("entry_price")

    def update_pairs(self, new_pairs):
        """
        Met à jour dynamiquement la liste des paires et réinitialise PPO avec le bon input_dim.
        """
        self.pairs_valid = new_pairs
        self._initialize_ai()  # Recrée PPO et l'input_dim pour les nouvelles paires

    def check_short_stop(self, symbol, price: float = None, trailing_pct: float = 0.03):
        """
        Déclenche le stop-loss court et/ou le trailing stop sur une position short BingX.
        - trailing_pct : trailing stop en % (ex: 0.03 = 3%)
        """
        pos = self.positions.get(symbol)
        if not pos or pos.get("side") != "short":
            return False
        entry = pos.get("entry_price")
        if entry is None:
            return False

        # Récupère le prix courant si non fourni
        if price is None:
            try:
                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                ticker = self.bingx_client.fetch_ticker_sync(symbol_bingx)
                price = float(ticker["last"])
            except Exception:
                return False

        # Initialisation du plus bas atteint depuis l'ouverture
        if "min_price" not in pos or pos["min_price"] is None:
            pos["min_price"] = price

        # Màj du plus bas (pour trailing stop)
        if price < pos["min_price"]:
            pos["min_price"] = price

        # Stop-loss court (si perte trop forte = prix monte trop)
        if price > entry * (1 + self.stop_loss_pct):
            self.logger.warning(
                f"[SHORT STOPLOSS] Déclenché sur {symbol}: prix={price} > {entry} + {self.stop_loss_pct*100}%"
            )
            return True

        # Trailing stop (si le prix remonte de X% par rapport au plus bas atteint)
        if price > pos["min_price"] * (1 + trailing_pct):
            self.logger.warning(
                f"[SHORT TRAILING STOP] Déclenché sur {symbol}: prix={price} > min={pos['min_price']} + {trailing_pct*100}%"
            )
            return True

        return False

    def update_pairs_from_config(self):
        self.pairs_valid = self.config["TRADING"]["pairs"]

        self.ws_collector = BufferedWSCollector(
            symbols=[s.replace("/", "").upper() for s in self.pairs_valid],
            timeframes=self.config["TRADING"]["timeframes"],
            maxlen=2000,
        )
        self.env = TradingEnv(
            trading_pairs=self.pairs_valid,
            timeframes=self.config["TRADING"]["timeframes"],
        )
        self._initialize_ai()

    def get_ws_orderbook(self, symbol):
        """
        Récupère le carnet d'ordres (bid/ask) depuis le ws_collector (WebSocket) ou via Binance API REST en fallback.
        - symbol : exemple 'BTCUSDC'
        Retourne : tuple (bid, ask) ou (None, None) si non dispo.
        """
        try:
            # Essai WebSocket
            if hasattr(self, "ws_collector") and self.ws_collector is not None:
                bid, ask = self.ws_collector.get_orderbook(symbol)
                # Si les valeurs existent et sont numériques, retourne-les
                if bid is not None and ask is not None:
                    return float(bid), float(ask)
            # Fallback sur Binance API REST
            if (
                getattr(self, "is_live_trading", False)
                and hasattr(self, "binance_client")
                and self.binance_client is not None
            ):
                try:
                    ob = self.binance_client.get_order_book(symbol=symbol, limit=5)
                    best_bid = float(ob["bids"][0][0]) if ob["bids"] else None
                    best_ask = float(ob["asks"][0][0]) if ob["asks"] else None
                    print("[FALLBACK REST] Carnet d'ordres récupéré via REST Binance.")
                    return best_bid, best_ask
                except Exception as e:
                    self.logger.warning(
                        f"[WS] Erreur récupération orderbook Binance API pour {symbol}: {e}"
                    )
        except Exception as e:
            self.logger.warning(
                f"[WS] Erreur récupération orderbook WS pour {symbol}: {e}"
            )
        return None, None

    async def execute_arbitrage_cross_exchange(self, opportunity, amount):
        """
        Exécute un arbitrage spot cross-exchange réel avec gestion des erreurs, logs et notifications Telegram.
        Args:
            opportunity (dict): dict contenant buy_exchange, sell_exchange, symbol, buy_price, sell_price, etc.
            amount (float): montant à investir (en devise quote, ex USDC)
        """
        try:
            buy_exchange = self.brokers[opportunity["buy_exchange"]]
            sell_exchange = self.brokers[opportunity["sell_exchange"]]
            symbol = opportunity["symbol"]
            base_currency = symbol.split("/")[0]
            quote_currency = symbol.split("/")[1]

            # 1. Vérification du solde disponible
            balance = await buy_exchange.fetch_balance()
            available = balance[quote_currency]["free"]
            if available < amount:
                msg = f"❌ Solde insuffisant sur {opportunity['buy_exchange']} ({available} {quote_currency} < {amount})"
                log_dashboard(msg)
                await self.telegram.send_message(msg)
                return {"status": "error", "step": "balance", "message": msg}

            # 2. Achat sur buy_exchange
            buy_qty = round(amount / opportunity["buy_price"], 6)
            log_dashboard(
                f"🔄 Achat {buy_qty} {base_currency} sur {opportunity['buy_exchange']} @ {opportunity['buy_price']}"
            )
            await self.telegram.send_message(
                f"🔄 Achat {buy_qty} {base_currency} sur {opportunity['buy_exchange']} @ {opportunity['buy_price']}"
            )
            buy_order = await buy_exchange.create_order(
                symbol=symbol, type="market", side="buy", amount=buy_qty
            )
            log_dashboard(f"✅ Ordre d'achat passé: {buy_order}")
            await self.telegram.send_message(
                f"✅ Ordre d'achat passé sur {opportunity['buy_exchange']}: {buy_order.get('id','?')}"
            )

            # 3. Retrait vers sell_exchange
            deposit_address = await sell_exchange.fetch_deposit_address(base_currency)
            withdrawal_fee = self.config["withdrawal_fees"][
                opportunity["buy_exchange"]
            ][base_currency]
            transfer_amount = buy_qty - withdrawal_fee
            if transfer_amount <= 0:
                msg = f"❌ Montant à transférer insuffisant (après frais: {transfer_amount} {base_currency})"
                log_dashboard(msg)
                await self.telegram.send_message(msg)
                return {"status": "error", "step": "withdraw", "message": msg}

            log_dashboard(
                f"🔄 Retrait {transfer_amount} {base_currency} vers {deposit_address['address']} ({opportunity['sell_exchange']})"
            )
            await self.telegram.send_message(
                f"🔄 Retrait {transfer_amount} {base_currency} vers {deposit_address['address']} ({opportunity['sell_exchange']})"
            )
            withdraw_result = await buy_exchange.withdraw(
                code=base_currency,
                amount=transfer_amount,
                address=deposit_address["address"],
            )
            log_dashboard(f"✅ Retrait initié: {withdraw_result}")
            await self.telegram.send_message(
                f"✅ Retrait initié: {withdraw_result.get('id','?')}"
            )

            # 4. Attente confirmation dépôt sur sell_exchange
            poll_interval = 30
            max_wait = 1800
            waited = 0
            while waited < max_wait:
                deposits = await sell_exchange.fetch_deposits(code=base_currency)
                if any(
                    d.get("amount", 0) >= transfer_amount and d.get("status") == "ok"
                    for d in deposits
                ):
                    log_dashboard(
                        f"✅ Dépôt confirmé sur {opportunity['sell_exchange']}"
                    )
                    await self.telegram.send_message(
                        f"✅ Dépôt confirmé sur {opportunity['sell_exchange']}"
                    )
                    break
                await asyncio.sleep(poll_interval)
                waited += poll_interval
            else:
                msg = (
                    f"❌ Timeout confirmation dépôt sur {opportunity['sell_exchange']}"
                )
                log_dashboard(msg)
                await self.telegram.send_message(msg)
                return {"status": "error", "step": "deposit", "message": msg}

            # 5. Vente sur sell_exchange
            log_dashboard(
                f"🔄 Vente {transfer_amount} {base_currency} sur {opportunity['sell_exchange']} @ {opportunity['sell_price']}"
            )
            await self.telegram.send_message(
                f"🔄 Vente {transfer_amount} {base_currency} sur {opportunity['sell_exchange']} @ {opportunity['sell_price']}"
            )
            sell_order = await sell_exchange.create_order(
                symbol=symbol, type="market", side="sell", amount=transfer_amount
            )
            log_dashboard(f"✅ Ordre de vente passé: {sell_order}")
            await self.telegram.send_message(
                f"✅ Ordre de vente passé sur {opportunity['sell_exchange']}: {sell_order.get('id','?')}"
            )

            # 6. Calcul du profit réel
            initial_value = amount
            final_value = sell_order.get(
                "cost", transfer_amount * opportunity["sell_price"]
            )
            profit = final_value - initial_value
            msg = f"💰 Arbitrage terminé sur {symbol}: Profit net {profit:.2f} {quote_currency}"
            log_dashboard(msg)
            await self.telegram.send_message(msg)

            return {
                "status": "success",
                "profit": profit,
                "buy_order": buy_order,
                "sell_order": sell_order,
                "transfer_amount": transfer_amount,
            }

        except Exception as e:
            msg = f"❌ Erreur arbitrage cross-exchange: {str(e)}"
            log_dashboard(msg)
            await self.telegram.send_message(msg)
            return {"status": "error", "step": "exception", "message": str(e)}

    async def test_news_sentiment(self):
        """
        Test manuel du batch d'analyse de sentiment des news.
        Exécute l'analyse Bert/FinBERT sur toutes les news du buffer et affiche le résumé global.
        """
        news = await self.news_analyzer.fetch_all_news()
        results = self.news_analyzer.analyze_sentiment_batch(news)
        summary = self.news_analyzer.get_sentiment_summary()
        print("Sentiment summary:", summary)

    def check_reload_dl_model(self):
        path = "src/models/cnn_lstm_model.pth"
        if os.path.exists(path):
            if self.dl_model is None:
                print(
                    "[ERROR] Modèle IA non initialisé, impossible de charger les poids."
                )
                return
            mtime = os.path.getmtime(path)
            if self.dl_model_last_mtime is None or mtime > self.dl_model_last_mtime:
                self.dl_model.load_weights(path)
                self.ai_enabled = self.dl_model is not None
                self.dl_model_last_mtime = mtime
                print(f"♻️ Nouveau modèle IA chargé automatiquement ({path})")

    async def _news_analysis_loop(self):
        """
        Boucle d’analyse des news avec pause automatique intelligente.
        Déclenche la pause selon sentiment, impact, classification, multi-source, volatilité, etc.
        """
        log_dashboard("[NEWS] Lancement boucle d'analyse des news…")
        while True:
            try:
                if not self.news_enabled or not self.news_analyzer:
                    await asyncio.sleep(self.news_update_interval)
                    continue

                self.logger.info("Fetching latest news for sentiment analysis")
                news_data = await self.news_analyzer.fetch_all_news()
                news_data = self.enrich_news_symbols(
                    news_data
                )  # Ajout des symboles aux news

                sentiment_analysis = {}
                try:
                    sentiment_analysis = await self.news_analyzer.update_analysis()
                except Exception:
                    self.logger.error("Erreur update_analysis", exc_info=True)
                    # sentiment_analysis reste {}

                # Extraction des items analysés
                sentiment_scores = (
                    sentiment_analysis.get("items", [])
                    if isinstance(sentiment_analysis, dict)
                    else []
                )

                # MAJ des données de sentiment dans le bot
                try:
                    await self._update_sentiment_data(sentiment_scores)
                except Exception:
                    pass

                # Sauvegarde dans shared_data.json
                try:
                    await self._save_sentiment_data(sentiment_scores, news_data)
                except Exception as e:
                    self.logger.error(f"Erreur lors de la sauvegarde du sentiment: {e}")

                # Envoi du résumé des news sur Telegram
                try:
                    await self.telegram.send_news_summary(news_data[:5])
                except Exception:
                    pass

                # === INTÉGRATION PAUSE INTELLIGENTE ===
                # Pour chaque news, analyse le besoin de pause
                for news in news_data:
                    pause_decision = self.news_pause_manager.should_pause(
                        news, self.market_data
                    )
                    if pause_decision:
                        self.news_pause_manager.activate_pause(pause_decision)
                        log_dashboard(
                            f"🚨 Pause déclenchée automatique: {pause_decision}"
                        )
                        # Optionnel: notification Telegram
                        try:
                            await self.telegram.send_message(
                                f"🚨 Pause automatique déclenchée\n"
                                f"Type: {pause_decision.get('type')}\n"
                                f"Paire: {pause_decision.get('pair', 'Toutes')}\n"
                                f"Raison: {pause_decision.get('reason')}\n"
                                f"Durée: {pause_decision.get('duration', 'N/A')} cycles"
                            )
                        except Exception:
                            pass

                # === LOG SENTIMENT GLOBAL ===
                try:
                    with open(self.data_file, "r") as f:
                        shared_data = json.load(f)
                    sentiment_data = shared_data.get("sentiment", {})
                    avg_sentiment = sentiment_data.get("overall_sentiment", 0)
                    impact_score = sentiment_data.get("impact_score", 0)
                    major_events = sentiment_data.get("major_events", "")

                    log_dashboard(
                        f"[NEWS] Score sentiment global: {avg_sentiment:.2f} | Impact: {impact_score:.2f} | Événements: {major_events}"
                    )
                except Exception as e:
                    print(
                        f"[NEWS] Impossible d'afficher le score sentiment global: {e}"
                    )

            except Exception as e:
                self.logger.error(f"News analysis error: {e}")

            await asyncio.sleep(self.news_update_interval)

    async def analyze_signals(self, symbol, ohlcv_df, indicators, tf="1h"):
        """
        Analyse complète des signaux de trading avec :
        - Analyse technique avancée avec 12+ indicateurs
        - Intégration IA et apprentissage profond
        - Sentiment des news et impact marché
        - Analyse volume, liquidité et orderflow
        - Structure de marché et niveaux clés
        - Fusion pondérée intelligente des signaux
        - Multi-timeframe confirmation
        - Monitoring système et performance
        """

        def is_valid(val):
            return val is not None and not (
                isinstance(val, float) and (np.isnan(val) or np.isinf(val))
            )

        try:
            # === 1. Monitoring système ===
            system_metrics = (
                self.monitor_system_health()
                if hasattr(self, "monitor_system_health")
                else {}
            )
            if (
                system_metrics.get("cpu_usage", 0) > 90
                or system_metrics.get("memory_usage", 0) > 90
            ):
                log_dashboard(
                    f"⚠️ Surcharge système détectée sur {symbol}, réduction risque!"
                )
                risk_multiplier = 0.5
            else:
                risk_multiplier = 1.0

            # === 2. Validation et logs initiaux ===
            print(
                f"[DEBUG OHLVC] {symbol} {tf} close: {ohlcv_df['close'].tail(5).tolist() if 'close' in ohlcv_df else 'NO CLOSE'}"
            )
            print(f"[DEBUG INDICATORS] {symbol} {tf}: {indicators}")

            # === 3. Configuration stratégie et paramètres ===
            params = getattr(self, "signal_fusion_params", None)
            if hasattr(self, "auto_strategy_config") and self.auto_strategy_config:
                log_dashboard(f"[STRATEGY] Stratégie AUTO-GÉNÉRÉE pour {symbol}")
                auto_cfg = self.auto_strategy_config
            else:
                log_dashboard(f"[STRATEGY] Stratégie STANDARD")

            # === 4. Extraction et validation des indicateurs ===
            indics = {
                "close": (
                    ohlcv_df["close"].iloc[-1]
                    if "close" in ohlcv_df and len(ohlcv_df) > 0
                    else None
                ),
                "prev_close": (
                    ohlcv_df["close"].iloc[-2]
                    if "close" in ohlcv_df and len(ohlcv_df) >= 2
                    else None
                ),
                "open": (
                    ohlcv_df["open"].iloc[-1]
                    if "open" in ohlcv_df and len(ohlcv_df) > 0
                    else None
                ),
                "high": (
                    ohlcv_df["high"].iloc[-1]
                    if "high" in ohlcv_df and len(ohlcv_df) > 0
                    else None
                ),
                "low": (
                    ohlcv_df["low"].iloc[-1]
                    if "low" in ohlcv_df and len(ohlcv_df) > 0
                    else None
                ),
                "volume": (
                    ohlcv_df["volume"].iloc[-1]
                    if "volume" in ohlcv_df and len(ohlcv_df) > 0
                    else None
                ),
                "sma_20": indicators.get("sma_20"),
                "sma_50": indicators.get("sma_50"),
                "ema_20": indicators.get("ema_20"),
                "rsi_14": indicators.get("rsi_14"),
                "macd": indicators.get("macd"),
                "macd_signal": indicators.get("macd_signal"),
                "macd_hist": indicators.get("macd_hist"),
                "bb_upper": indicators.get("bb_upper"),
                "bb_lower": indicators.get("bb_lower"),
                "psar": indicators.get("psar"),
                "momentum_10": indicators.get("momentum_10"),
                "zscore_20": indicators.get("zscore_20"),
                "stochrsi": indicators.get("stochrsi"),
                "kc_upper": indicators.get("kc_upper"),
                "kc_lower": indicators.get("kc_lower"),
                "vwap": indicators.get("vwap"),
            }

            # === 5. Analyses avancées ===
            # Volume et liquidité
            volume_profile = self.analyze_volume_profile(symbol, tf)
            order_pressure = self.analyze_order_pressure(symbol)

            # Structure de marché
            market_struct = self.identify_market_structure(ohlcv_df)

            # Momentum et volatilité
            squeeze_data = self.calculate_squeeze_momentum(ohlcv_df)

            # Order flow et delta volume
            flow_analysis = self.analyze_order_flow(ohlcv_df)

            # === 6. Score technique ===
            tech_score = 0
            tech_factors = 0

            # Moyennes mobiles
            for ma_type, (price, weight) in {
                "sma_20": (indics["sma_20"], 2.0),
                "sma_50": (indics["sma_50"], 1.5),
                "ema_20": (indics["ema_20"], 2.5),
            }.items():
                if is_valid(indics["close"]) and is_valid(price):
                    tech_factors += 1
                    pct_diff = (indics["close"] - price) / price * 100
                    tech_score += np.clip(pct_diff * weight, -1, 1)

            # RSI
            if is_valid(indics["rsi_14"]):
                tech_factors += 1
                if indics["rsi_14"] > 70:
                    tech_score -= 0.8
                elif indics["rsi_14"] < 30:
                    tech_score += 0.8
                else:
                    tech_score += (indics["rsi_14"] - 50) / 25

            # MACD
            if is_valid(indics["macd"]) and is_valid(indics["macd_signal"]):
                tech_factors += 1
                macd_diff = indics["macd"] - indics["macd_signal"]
                tech_score += np.clip(macd_diff * 10, -1, 1)

            if is_valid(indics["macd_hist"]):
                tech_factors += 1
                tech_score += np.clip(indics["macd_hist"] * 15, -1, 1)

            # Bollinger Bands
            if all(is_valid(indics[x]) for x in ["bb_upper", "bb_lower", "close"]):
                tech_factors += 1
                bb_position = (indics["close"] - indics["bb_lower"]) / (
                    indics["bb_upper"] - indics["bb_lower"]
                )
                if bb_position < 0.2:
                    tech_score += 0.6
                elif bb_position > 0.8:
                    tech_score -= 0.6

            # PSAR
            if all(
                is_valid(x)
                for x in [indics["psar"], indics["prev_close"], indics["close"]]
            ):
                tech_factors += 1
                if (
                    indics["prev_close"] < indics["psar"]
                    and indics["close"] > indics["psar"]
                ):
                    tech_score += 0.8
                elif (
                    indics["prev_close"] > indics["psar"]
                    and indics["close"] < indics["psar"]
                ):
                    tech_score -= 0.8

            # Momentum et Z-Score
            if is_valid(indics["momentum_10"]) and is_valid(indics["close"]):
                tech_factors += 1
                momentum_pct = indics["momentum_10"] / indics["close"] * 100
                tech_score += np.clip(momentum_pct * 5, -1, 1)

            if is_valid(indics["zscore_20"]):
                tech_factors += 1
                tech_score += np.clip(indics["zscore_20"] * 0.5, -1, 1)

            # Normalisation score technique
            if tech_factors > 0:
                tech_score = tech_score / tech_factors
                if abs(tech_score) > 0.3:
                    tech_score *= 1.2

            # === 7. Scores avancés ===
            # Volume Profile Score
            volume_score = 0
            if volume_profile:
                volume_score = (
                    0.3
                    if volume_profile.get("is_accumulation")
                    else -0.3 if volume_profile.get("is_distribution") else 0
                )
                if volume_profile.get("buy_volume_ratio", 0.5) > 0.6:
                    tech_score *= 1.2

            # Order Pressure Score
            pressure_score = 0
            if order_pressure:
                pressure_score = order_pressure.get(
                    "imbalance", 0
                ) * order_pressure.get("pressure_ratio", 1)

            # Market Structure Score
            structure_score = 0
            if market_struct:
                structure_score = market_struct.get("trend_strength", 0)
                if market_struct.get("volatility", 0) > 0.8:
                    tech_score *= 0.8

            # Squeeze Momentum Score
            squeeze_score = 0
            if squeeze_data and squeeze_data.get("squeeze_on"):
                squeeze_score = 0.3 if squeeze_data.get("momentum", 0) > 0 else -0.3

            # === 8. Score IA ===
            ai_score = 0
            if self.ai_enabled and hasattr(self, "dl_model") and self.dl_model:
                try:
                    features = await self._prepare_features_for_ai(symbol)
                    if features is not None:
                        ai_score = float(self.dl_model.predict(features))
                except Exception as e:
                    self.logger.warning(f"Erreur IA: {e}")

            # === 9. Score Sentiment ===
            sentiment_score = 0
            pair_key = symbol.replace("/", "").upper()
            if getattr(self, "news_enabled", False) and hasattr(self, "news_analyzer"):
                try:
                    sentiment_score = await self.news_analyzer.get_symbol_sentiment(
                        pair_key
                    )
                    if sentiment_score == 0:
                        sentiment_score = (
                            self.news_analyzer.get_sentiment_summary().get(
                                "sentiment_global", 0.0
                            )
                        )
                except Exception as e:
                    self.logger.error(f"Erreur sentiment {pair_key}: {e}")

            # === 10. Fusion pondérée des signaux ===
            signals = {
                "technical": tech_score,
                "ai": ai_score,
                "sentiment": sentiment_score,
                "volume": volume_score,
                "pressure": pressure_score,
                "structure": structure_score,
                "squeeze": squeeze_score,
            }

            weights = {
                "technical": 0.3,
                "ai": 0.2,
                "sentiment": 0.15,
                "volume": 0.15,
                "pressure": 0.1,
                "structure": 0.05,
                "squeeze": 0.05,
            }

            total_score = sum(
                score * weights[signal_type] for signal_type, score in signals.items()
            )
            total_score *= risk_multiplier  # Ajustement selon charge système

            # === 11. Multi-timeframe confirmation ===
            if tf in ["1h", "4h"]:
                try:
                    mtp_analysis = self.multi_timeframe_analysis(
                        symbol, ["15m", "1h", "4h"]
                    )
                    confirmation = self.calculate_confirmation_score(
                        {
                            "market_structure": market_struct,
                            "volume_profile": volume_profile,
                            "squeeze_momentum": squeeze_data,
                            "order_flow": order_pressure,
                        },
                        mtp_analysis,
                    )

                    if confirmation > 0.8:
                        total_score *= 1.2
                    elif confirmation < 0.2:
                        total_score *= 0.8

                except Exception as e:
                    self.logger.error(f"Erreur confirmation multi-TF: {e}")

            # === 12. Décision finale ===
            volatility = market_struct.get("volatility", 1) if market_struct else 1
            buy_threshold = params.get("buy_threshold", 0.2) * (1 + volatility)
            sell_threshold = params.get("sell_threshold", -0.2) * (1 + volatility)

            decision = {
                "action": "neutral",
                "confidence": abs(total_score),
                "signals": signals,
                "metrics": {
                    "volume_profile": volume_profile,
                    "order_pressure": order_pressure,
                    "market_structure": market_struct,
                    "squeeze": squeeze_data,
                    "system_health": system_metrics,
                    "confirmation_score": (
                        confirmation if "confirmation" in locals() else None
                    ),
                },
            }

            if total_score > buy_threshold:
                decision["action"] = "buy"
            elif total_score < sell_threshold:
                decision["action"] = "sell"

            # === 13. Logs détaillés ===
            log_msg = (
                f"[ANALYZE] {symbol} {tf} |\n"
                f"Tech: {tech_score:.3f} | AI: {ai_score:.3f} | Sent: {sentiment_score:.3f}\n"
                f"Vol: {volume_score:.3f} | Press: {pressure_score:.3f} | Struct: {structure_score:.3f}\n"
                f"Total: {total_score:.3f} | {decision['action'].upper()} ({decision['confidence']:.3f})"
            )
            log_dashboard(log_msg)
            print(f"[DEBUG] {log_msg}")

            # === 14. Backup automatique (tous les 100 cycles) ===
            if hasattr(self, "current_cycle") and self.current_cycle % 100 == 0:
                self.backup_critical_data()

            return decision

        except Exception as e:
            self.logger.error(f"Erreur analyze_signals: {e}")
            return {"action": "neutral", "confidence": 0, "signals": {}}

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
        """
        Détecte les opportunités d'arbitrage cross-quote USDC/USDT :
        Compare par exemple BTC/USDC sur Binance à BTC/USDT sur les autres brokers,
        avec adaptation du format des symboles selon chaque broker.
        """

        def get_broker_symbol(coin, quote, broker):
            # Adapter au format attendu par chaque broker
            if broker == "binance":
                return f"{coin}{quote}"
            elif broker in ["okx", "bingx"]:
                return f"{coin}-{quote}"
            elif broker == "gateio":
                return f"{coin}_{quote}"
            elif broker == "blofin":
                return f"{coin}{quote}"  # Si différent, adapter ici !
            else:
                return f"{coin}/{quote}"  # Fallback

        if not self.is_live_trading:
            log_dashboard("[ARBITRAGE] Pas en mode live trading, détection annulée.")
            return []
        log_dashboard("[ARBITRAGE] Démarrage détection arbitrage USDC/USDT…")
        opportunities = []
        pairs_to_check = [pair] if pair else self.pairs_valid
        MIN_PROFIT_THRESHOLD = 0.15
        MIN_VOLUME_USD = 0
        MAX_SPREAD = 0.5

        try:
            for current_pair in pairs_to_check:
                try:
                    coin = current_pair.split("/")[0]
                    # Symboles pour chaque broker
                    binance_symbol = get_broker_symbol(coin, "USDC", "binance")

                    # Prix Binance (USDC)
                    binance_ticker = self.binance_client.get_ticker(
                        symbol=binance_symbol
                    )
                    binance_price = float(binance_ticker.get("lastPrice") or 0)
                    binance_volume = float(binance_ticker.get("volume", 0))

                    # Liste des brokers à comparer (USDT)
                    exchanges_to_check = [
                        {"name": "okx", "client": self.brokers.get("okx")},
                        {"name": "gateio", "client": self.brokers.get("gateio")},
                        {"name": "blofin", "client": self.brokers.get("blofin")},
                        {"name": "bingx", "client": self.brokers.get("bingx")},
                    ]

                    for exchange in exchanges_to_check:
                        broker = exchange["name"]
                        binance_symbol = get_broker_symbol(coin, "USDC", "binance")
                        other_symbol = get_broker_symbol(coin, "USDT", broker)
                        if not exchange["client"]:
                            continue

                        try:
                            other_symbol = get_broker_symbol(coin, "USDT", broker)
                            # Récupération du prix sur l'autre broker (USDT)
                            ticker = await exchange["client"].fetch_ticker(other_symbol)
                            exchange_price = float(ticker["last"])
                            if not exchange_price or not binance_price:
                                continue

                            # Calcul du spread
                            price_diff = exchange_price - binance_price
                            profit_pct = (
                                (price_diff / binance_price) * 100
                                if binance_price > 0
                                else 0
                            )

                            # Opportunité cross-quote
                            if profit_pct > MIN_PROFIT_THRESHOLD:
                                opportunity = {
                                    "pair": coin,
                                    "exchange1": "Binance (USDC)",
                                    "exchange2": f"{broker} (USDT)",
                                    "price1": binance_price,
                                    "price2": exchange_price,
                                    "diff_percent": profit_pct,
                                    "volume_24h": binance_volume * binance_price,
                                    "estimated_profit": profit_pct - 0.2,  # Après frais
                                    "route": f"Buy {coin}/USDC (Binance) -> Transfer {coin} -> Sell {coin}/USDT ({broker})",
                                }
                                log_dashboard(
                                    f"[ARBITRAGE] OPPORTUNITÉ: {coin}: {binance_price} (Binance USDC) <> {exchange_price} ({broker} USDT) | Diff: {profit_pct:.2f}%"
                                )
                                opportunities.append(opportunity)
                                self.logger.info(
                                    f"Opportunité d'arbitrage cross-quote détectée pour {coin}: {opportunity}"
                                )

                        except Exception as e:
                            print(f"[ARBITRAGE] Erreur sur {broker}: {e}")
                            self.logger.error(f"Erreur sur {broker}: {e}")
                            continue

                except Exception as e:
                    print(
                        f"[ARBITRAGE] Erreur lors du traitement de {current_pair}: {e}"
                    )
                    self.logger.error(
                        f"Erreur lors du traitement de {current_pair}: {e}"
                    )
                    continue

            if opportunities:
                print(
                    f"[ARBITRAGE] {len(opportunities)} opportunités détectées ce cycle."
                )
            else:
                print("[ARBITRAGE] Aucune opportunité détectée ce cycle.")

            return opportunities

        except Exception as e:
            print(f"[ARBITRAGE] Erreur globale détection arbitrage: {e}")
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
        """Initialise les composants d'IA et du trading live Binance"""
        try:
            log_dashboard("Initialisation des modèles d'IA...")
            if not self.env:
                raise ValueError("L'environnement de trading n'est pas initialisé")

            # 1. Constantes IA
            self.N_FEATURES = 8
            self.N_STEPS = 63

            # 2. Hyperparams AutoML si dispo
            hp_path = "config/best_hyperparams.json"
            if os.path.exists(hp_path):
                with open(hp_path, "r") as f:
                    best_hp = json.load(f)
                self.config["AI"].update(best_hp)
                print(f"[AI] Hyperparams optimisés chargés depuis {hp_path}: {best_hp}")
            else:
                print(
                    "[AI] Pas d'hyperparams optimisés trouvés, utilisation des valeurs par défaut."
                )

            # 3. Deep Learning Model
            self.dl_model = DeepLearningModel()
            self.dl_model.initialize()
            weights_path = "src/models/cnn_lstm_model.pth"
            if os.path.exists(weights_path):
                self.dl_model.load_weights(weights_path)
                print(f"[DL] Modèle chargé depuis {weights_path}")
            else:
                print(
                    f"[DL WARNING] Aucun modèle entraîné trouvé à {weights_path} ! Prédictions IA non fiables."
                )
            if os.path.exists(weights_path):
                self.dl_model_last_mtime = os.path.getmtime(weights_path)
            else:
                self.dl_model_last_mtime = None
            print(
                f"[DEBUG] paires_valid utilisées IA: {self.pairs_valid} (count={len(self.pairs_valid)})"
            )

            # 4. PPO
            input_dim = self.get_input_dim()
            num_pairs = len(self.pairs_valid)
            env_config = {
                "env": self.env,
                "input_dim": input_dim,
                "learning_rate": self.config["AI"]["learning_rate"],
                "batch_size": self.config["AI"]["batch_size"],
                "n_epochs": self.config["AI"]["n_epochs"],
                "verbose": 1,
            }
            self.ppo_strategy = PPOStrategy(env_config)
            if self.ppo_strategy.model is None:
                raise ValueError("Échec de l'initialisation du modèle PPO")
            self.ai_enabled = True
            log_dashboard("✅ Modèles d'IA initialisés avec succès")
        except Exception as e:
            print(f"❌ Erreur initialisation IA: {str(e)}")
            self.ai_enabled = False
            self.dl_model = None
            self.ppo_strategy = None

        # 5. Telegram & Logger
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.last_telegram_update = datetime.utcnow()
        self.logger = logger

        # 6. Initialisation de l'API Binance (live/simu)
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        if self.api_key and self.api_secret:
            self.binance_client = Client(self.api_key, self.api_secret)
            self.binance_connector = BinanceConnector()
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
        print(f"[DEBUG] is_live_trading après init: {self.is_live_trading}")

        # 7. PPO (recheck, for redundancy)
        try:
            print("Configuration de la stratégie PPO...")
            N_FEATURES = self.N_FEATURES
            N_STEPS = self.N_STEPS
            num_pairs = len(self.pairs_valid)
            env_config = {
                "env": self.env,
                "input_dim": N_FEATURES * N_STEPS * num_pairs,
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_epochs": 10,
                "verbose": 1,
            }
            if not hasattr(self.env, "reset") or not hasattr(self.env, "step"):
                raise ValueError("Trading environment missing required methods")
            self.ppo_strategy = PPOStrategy(env_config)
            if self.ppo_strategy.model is None:
                raise ValueError("PPO model failed to initialize")
            log_dashboard("✅ PPO Strategy initialized successfully")
        except Exception as e:
            print(f"❌ Erreur initialisation PPO: {str(e)}")
            self.ppo_strategy = None

        # 8. Sentiment Analyzer
        try:
            self.news_analyzer = NewsSentimentAnalyzer(self.config)
            self.news_enabled = True
            self.dl_model_last_mtime = None
            self.news_weight = 0.2
            self.news_update_interval = 300
            self.logger.info("News sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize news analyzer: {e}")
            self.news_enabled = False
            self.news_analyzer = None

        log_dashboard(f"✅ Bot initialisé avec Telegram: {bool(TELEGRAM_BOT_TOKEN)}")
        log_dashboard(f"✅ Trading en direct: {self.is_live_trading}")
        log_dashboard(f"✅ IA activée: {self.ai_enabled}")
        log_dashboard(f"✅ Analyse de news activée: {self.news_enabled}")

    # Ajoute la méthode get_input_dim à ta classe TradingBotM4 si ce n'est pas déjà fait :
    def get_input_dim(self):
        return self.N_FEATURES * self.N_STEPS * len(self.pairs_valid)

    async def test_news_sentiment(self):
        """
        Test manuel du batch d'analyse de sentiment des news.
        Exécute l'analyse Bert/FinBERT sur toutes les news du buffer et affiche le résumé global.
        """
        news = await self.news_analyzer.fetch_all_news()
        results = self.news_analyzer.analyze_sentiment_batch(news)
        summary = self.news_analyzer.get_sentiment_summary()
        print("Sentiment summary:", summary)

    def check_stop_loss(self, symbol, price: float = None):
        """
        Stop-loss dynamique basé sur la volatilité (ATR).
        """
        try:
            pos = self.positions.get(symbol)
            if not pos or pos.get("side") != "long":
                return False
            entry = pos.get("entry_price")
            if entry is None:
                return False

            symbol_ws = symbol.replace("/", "").upper()
            if price is None:
                price = self.ws_collector.get_last_price(symbol_ws)
                if (
                    price is None
                    and symbol_ws in self.market_data
                    and "1h" in self.market_data[symbol_ws]
                ):
                    closes = self.market_data[symbol_ws]["1h"].get("close", [])
                    if closes:
                        price = closes[-1]
            if price is None:
                return False

            # Calcul ATR sur 1h pour stop dynamique
            df_ohlcv = pd.DataFrame(self.market_data[symbol_ws]["1h"])
            atr = calculate_atr(df_ohlcv, period=14)
            dynamic_stop_pct = max(0.01, min(atr / entry, 0.10))  # Entre 1% et 10% max

            loss = (price - entry) / entry
            if loss < -dynamic_stop_pct:
                print(
                    f"[STOPLOSS] Déclenché sur {symbol}: perte = {loss:.2%} (ATR dynamique={dynamic_stop_pct:.2%})"
                )
                return True
            return False
        except Exception as e:
            print(f"[STOPLOSS] Erreur vérification stop-loss: {e}")
            return False

    async def execute_trade(
        self, symbol, side, amount, price=None, iceberg=False, iceberg_visible_size=0.1
    ):
        """
        Exécute un ordre de trading avec logs détaillés.
        - BUY sur Binance spot (quoteOrderQty)
        - SELL sur Binance spot (revente, si déjà long OU si solde réel suffisant)
        - SHORT sur BingX (futures)
        - BUY sur BingX pour rachat short
        - Gère le suivi de position SPOT et le stop-loss automatique
        """

        if not self.is_live_trading:
            log_dashboard(
                f"[ORDER] SIMULATION: {side} {amount} {symbol} @ {price} (iceberg={iceberg})"
            )
            self.logger.info(
                f"SIMULATION: {side} {amount} {symbol} @ {price} (iceberg={iceberg})"
            )
            # Gestion état simulée
            if side.upper() == "BUY":
                if self.is_long(symbol):
                    log_dashboard(
                        f"[ORDER] Déjà long sur {symbol}, achat ignoré (simu)"
                    )
                    return {"status": "skipped", "reason": "already long"}
                self.positions[symbol] = {
                    "side": "long",
                    "entry_price": price or 0,
                    "amount": amount,
                }
            elif side.upper() == "SELL":
                if not self.is_long(symbol):
                    log_dashboard(
                        f"[ORDER] Pas en position long sur {symbol}, vente ignorée (simu)"
                    )
                    return {"status": "skipped", "reason": "not in position"}
                self.positions.pop(symbol, None)
            elif side.upper() == "SHORT":
                if self.is_short(symbol):
                    log_dashboard(
                        f"[ORDER] Déjà short sur {symbol}, short ignoré (simu)"
                    )
                    return {"status": "skipped", "reason": "already short"}
                self.positions[symbol] = {
                    "side": "short",
                    "entry_price": price or 0,
                    "amount": amount,
                    "min_price": price or 0,
                }
            elif side.upper() == "BUY" and self.is_short(symbol):
                if not self.is_short(symbol):
                    log_dashboard(
                        f"[ORDER] Pas en position short sur {symbol}, rachat ignoré (simu)"
                    )
                    return {"status": "skipped", "reason": "not in short"}
                self.positions.pop(symbol, None)
            return {
                "status": "simulated",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "iceberg": iceberg,
            }

        try:
            log_dashboard(
                f"[ORDER] Tentative d'exécution: {side} {amount} {symbol} (iceberg: {iceberg})"
            )

            # ----- ACHAT SPOT -----
            if side.upper() == "BUY" and symbol.endswith("USDC"):
                if self.is_long(symbol):
                    log_dashboard(f"[ORDER] Déjà long sur {symbol}, achat ignoré.")
                    return {"status": "skipped", "reason": "already long"}
                bid, ask = self.get_ws_orderbook(symbol)
                if bid is None or ask is None:
                    log_dashboard(
                        f"[ORDER] Orderbook WS non dispo pour {symbol}, annulation de l'ordre."
                    )
                    return {"status": "error", "reason": "Orderbook WS not available"}
                orderbook = {"bids": [[bid, 1.0]], "asks": [[ask, 1.0]]}
                recent_trades = []
                market_data = {
                    "recent_trades": recent_trades,
                    "volatility": self.calculate_volatility(
                        self.market_data.get(symbol, {}).get("1h", {})
                    ),
                    "regime": self.regime,
                    "binance_client": self.binance_client,
                }
                result = await self.executor.execute_order(
                    symbol=symbol,
                    side=side,
                    quoteOrderQty=amount,
                    orderbook=orderbook,
                    market_data=market_data,
                    iceberg=iceberg,
                    iceberg_visible_size=iceberg_visible_size,
                )
                if result.get("status") == "completed":
                    self.positions[symbol] = {
                        "side": "long",
                        "entry_price": result.get("avg_price", price),
                        "amount": result.get("filled_amount", amount),
                    }

            # ----- VENTE SPOT -----
            elif side.upper() == "SELL" and symbol.endswith("USDC"):
                # 1. Vente si position virtuelle "long"
                allow_sell = False
                use_amount = None
                if self.is_long(symbol):
                    allow_sell = True
                    use_amount = self.positions[symbol]["amount"]
                else:
                    # 2. Sinon, vente si solde réel Binance dispo
                    asset = symbol.replace("USDC", "")
                    balance = None
                    try:
                        balance = self.binance_client.get_asset_balance(asset=asset)
                    except Exception as e:
                        log_dashboard(
                            f"[ORDER] Erreur récupération balance {asset}: {e}"
                        )
                    if balance and float(balance.get("free", 0)) >= amount:
                        allow_sell = True
                        use_amount = amount
                        log_dashboard(
                            f"[ORDER] Vente autorisée sur solde réel {asset}: {balance['free']}"
                        )
                    else:
                        log_dashboard(
                            f"[ORDER] Pas en position long ni de solde suffisant sur {symbol}, vente ignorée."
                        )
                        return {
                            "status": "skipped",
                            "reason": "not in position or insufficient balance",
                        }

                bid, ask = self.get_ws_orderbook(symbol)
                if bid is None or ask is None:
                    log_dashboard(
                        f"[ORDER] Orderbook WS non dispo pour {symbol}, annulation de l'ordre."
                    )
                    return {"status": "error", "reason": "Orderbook WS not available"}
                orderbook = {"bids": [[bid, 1.0]], "asks": [[ask, 1.0]]}
                market_data = {
                    "recent_trades": [],
                    "volatility": self.calculate_volatility(
                        self.market_data.get(symbol, {}).get("1h", {})
                    ),
                    "regime": self.regime,
                    "binance_client": self.binance_client,
                }
                result = await self.executor.execute_order(
                    symbol=symbol,
                    side=side,
                    quoteOrderQty=use_amount,
                    orderbook=orderbook,
                    market_data=market_data,
                    iceberg=iceberg,
                    iceberg_visible_size=iceberg_visible_size,
                )
                if result.get("status") == "completed" and self.is_long(symbol):
                    self.positions.pop(symbol, None)

            # ----- OUVERTURE SHORT BINGX -----
            elif side.upper() == "SHORT":
                if self.is_short(symbol):
                    log_dashboard(f"[ORDER] Déjà short sur {symbol}, short ignoré.")
                    return {"status": "skipped", "reason": "already short"}
                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                ticker = await self.bingx_client.fetch_ticker(symbol_bingx)
                price_bingx = float(ticker["last"])
                qty = amount / price_bingx
                result = await self.bingx_executor.short_order(
                    symbol_bingx, qty, leverage=3
                )
                if result.get("status") == "completed":
                    self.positions[symbol] = {
                        "side": "short",
                        "entry_price": price_bingx,
                        "amount": qty,
                        "min_price": price_bingx,
                    }

            # ----- FERMETURE SHORT BINGX -----
            elif side.upper() == "BUY" and self.is_short(symbol):
                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                pos = self.positions[symbol]
                qty = pos["amount"]
                # Il faut avoir une méthode close_short_order côté BingXOrderExecutor, sinon utiliser un BUY ordinaire sur futures
                result = await self.bingx_executor.close_short_order(symbol_bingx, qty)
                if result.get("status") == "completed":
                    self.positions.pop(symbol, None)

            else:
                return {"status": "rejected", "reason": "unsupported side"}

            # ----- LOGS & NOTIF -----
            if result["status"] == "completed":
                log_dashboard(
                    f"[ORDER] Exécuté avec succès: {side} {result.get('filled_amount', amount)} {symbol} @ {result.get('avg_price', price)}"
                )
                self.logger.info(
                    f"Order executed: {side} {result.get('filled_amount', amount)} {symbol} @ {result.get('avg_price', price)}"
                )
                self._update_performance_metrics(result)
                iceberg_info = (
                    f"\n🧊 <b>Ordre Iceberg</b> ({result.get('n_suborders', '')} sous-ordres)"
                    if result.get("iceberg")
                    else ""
                )
                await self.telegram.send_message(
                    f"💰 <b>Ordre exécuté</b>\n"
                    f"📊 {side} {result.get('filled_amount', amount)} {symbol} @ {result.get('avg_price', price)}\n"
                    f"💵 Total: ${float(result.get('filled_amount', amount)) * float(result.get('avg_price', price) or 0):.2f}"
                    f"{iceberg_info}"
                )
            else:
                print(f"[ORDER] Echec d'exécution: {side} {amount} {symbol}")

            return result

        except BinanceAPIException as e:
            print(f"[ORDER] Binance API error: {e}")
            self.logger.error(f"Binance API error: {e}")
            await self.telegram.send_message(f"⚠️ Erreur API Binance: {e}")
            return {"status": "error", "reason": str(e)}
        except Exception as e:
            print(f"[ORDER] Execution error: {e}")
            self.logger.error(f"Execution error: {e}")
            return {"status": "error", "reason": str(e)}

    def _update_performance_metrics(self, trade_result):
        """Met à jour les métriques de performance après un trade réel"""
        try:
            self.safe_update_shared_data(
                {"bot_status": {"performance": performance}}, self.data_file
            )

            performance = data["bot_status"]["performance"]

            # Mise à jour des statistiques
            performance["total_trades"] += 1

            # Calcul du profit/perte
            filled_amount = float(trade_result["filled_amount"])
            avg_price = float(trade_result["avg_price"])
            side = trade_result["side"]  # <-- side est une string

            if side == "buy":
                # Pour un achat, on ne sait pas encore si c'est gagnant
                pass
            elif side == "sell":
                # Pour une vente, on peut calculer le profit par rapport au prix d'achat moyen
                entry_price = trade_result.get("entry_price", 0)
                if entry_price > 0:
                    profit_pct = (
                        (avg_price / entry_price - 1) * 100
                        if side == "sell"
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
            self.safe_update_shared_data(
                {"bot_status": {"performance": performance}}, self.data_file
            )

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _prepare_features_for_ai(self, symbol):
        """
        Prépare les features pour les modèles d'IA (adapté pour PPO et DL).
        ATTENTION: Retourne TOUJOURS un dict avec les clés
        'close', 'high', 'low', 'volume', 'rsi', 'macd', 'volatility'.
        Si besoin pour PPO, ajoute aussi 'vol_ratio'.
        """
        try:
            N_STEPS = self.N_STEPS

            ohlcv = self.market_data.get(symbol, {}).get("1h", {})
            if not ohlcv or not isinstance(ohlcv, dict) or "close" not in ohlcv:
                return None

            closes = np.array(ohlcv.get("close", []))
            highs = np.array(ohlcv.get("high", []))
            lows = np.array(ohlcv.get("low", []))
            volumes = np.array(ohlcv.get("volume", []))

            # --- Vérification stricte sur la taille
            if (
                len(closes) < N_STEPS
                or len(highs) < N_STEPS
                or len(lows) < N_STEPS
                or len(volumes) < N_STEPS
            ):
                return None

            closes = closes[-N_STEPS:]
            highs = highs[-N_STEPS:]
            lows = lows[-N_STEPS:]
            volumes = volumes[-N_STEPS:]

            # RSI (14)
            delta = np.diff(closes)
            gain = (delta > 0) * delta
            loss = (delta < 0) * -delta
            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0.001
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))

            # MACD: EMA12 - EMA26
            ema12 = np.mean(closes[-12:]) if len(closes) >= 12 else closes[-1]
            ema26 = np.mean(closes[-26:]) if len(closes) >= 26 else closes[-1]
            macd = ema12 - ema26

            # Volatility: std des returns
            if len(closes) >= N_STEPS:
                returns = np.diff(np.log(closes))
                volatility = float(np.std(returns[-14:])) if len(returns) >= 14 else 0
            else:
                volatility = 0

            avg_volume = np.mean(volumes) if np.mean(volumes) > 0 else 1
            vol_ratio = float(volumes[-1]) / avg_volume if avg_volume > 0 else 1
            vol_ratio = min(1, vol_ratio / 3)

            features = {
                "close": closes / closes[0],
                "high": highs / highs[0] if highs[0] > 0 else highs,
                "low": lows / lows[0] if lows[0] > 0 else lows,
                "volume": volumes / volumes[0] if volumes[0] > 0 else volumes,
                "rsi": float(rsi) / 100,
                "macd": float(macd) / 100,
                "volatility": float(volatility),
                "vol_ratio": float(vol_ratio),
            }

            # Correction NaN/inf
            for k in features:
                arr = features[k]
                if isinstance(arr, np.ndarray):
                    if np.isnan(arr).any() or np.isinf(arr).any():
                        print(f"[WARN] NaN/inf détecté dans {k}, correction appliquée")
                        features[k] = np.nan_to_num(arr)
                else:
                    if np.isnan(features[k]) or np.isinf(features[k]):
                        print(f"[WARN] NaN/inf détecté dans {k}, correction appliquée")
                        features[k] = float(np.nan_to_num(features[k]))

            required_keys = [
                "close",
                "high",
                "low",
                "volume",
                "rsi",
                "macd",
                "volatility",
            ]
            for k in required_keys:
                if k not in features:
                    self.logger.error(
                        f"[AI FEATURES] Clé manquante dans features : {k}"
                    )
                    return None
            for k in ["close", "high", "low", "volume"]:
                if not (
                    isinstance(features[k], np.ndarray)
                    and features[k].shape == (N_STEPS,)
                ):
                    self.logger.error(
                        f"[AI FEATURES] Mauvais shape pour {k}: {type(features[k])}, shape={getattr(features[k], 'shape', None)}"
                    )
                    return None
            for k in ["rsi", "macd", "volatility", "vol_ratio"]:
                if not isinstance(features[k], (int, float, np.floating, np.integer)):
                    self.logger.error(
                        f"[AI FEATURES] Mauvais type pour {k}: {type(features[k])}"
                    )
                    return None

            return features

        except Exception as e:
            self.logger.error(f"Error preparing AI features: {e}")
            return None

    async def _merge_signals(self, symbol, dl_prediction, ppo_action):
        try:
            # 1. Vérification et initialisation des poids
            if not hasattr(self, "ai_weight"):
                self.ai_weight = 0.4  # Valeur par défaut

            # Conversion robuste du ai_weight si nécessaire
            try:
                ai_weight = float(self.ai_weight)
            except (TypeError, ValueError):
                self.logger.error(
                    "ai_weight invalide, utilisation de la valeur par défaut 0.4"
                )
                ai_weight = 0.4

            technical_weight = 1.0 - ai_weight

            # 2. Initialisation des structures
            if symbol not in self.market_data:
                self.market_data[symbol] = {}

            default_signals = {"trend": 0.0, "momentum": 0.0, "volatility": 0.0}

            current_signals = self.market_data[symbol].get(
                "signals", default_signals.copy()
            )

            # 3. Fonction de conversion universelle
            def safe_float(value, context=""):
                """Convertit n'importe quelle entrée en float de manière sécurisée"""
                if isinstance(value, (float, int)):
                    return float(value)

                if isinstance(value, dict):
                    # Extraction depuis les dictionnaires
                    for key in ["value", "action", "score", "prediction", "weight"]:
                        if key in value:
                            try:
                                return float(value[key])
                            except (TypeError, ValueError):
                                continue

                    # Fallback: premier float trouvé
                    for v in value.values():
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            continue

                # Fallback final
                self.logger.warning(
                    f"Conversion impossible pour {context}, utilisation de 0.0"
                )
                return 0.0

            # 4. Conversion des entrées
            dl_value = safe_float(dl_prediction, "dl_prediction")
            ppo_value = safe_float(ppo_action, "ppo_action")
            ai_signal = dl_value * 0.7 + ppo_value * 0.3

            # 5. Nettoyage des signaux existants
            clean_signals = {
                k: safe_float(v, f"signal {k}")
                for k, v in current_signals.items()
                if k in default_signals
            }

            # 6. Fusion finale
            merged_signals = {
                k: (v * technical_weight + ai_signal * ai_weight)
                for k, v in clean_signals.items()
            }

            # 7. Sauvegarde des résultats
            self.market_data[symbol]["signals"] = merged_signals
            self.market_data[symbol]["ai_prediction"] = ai_signal

            return merged_signals

        except Exception as e:
            self.logger.error(f"ERREUR dans _merge_signals: {str(e)}", exc_info=True)
            return default_signals.copy()

    async def _update_sentiment_data(self, sentiment_scores):
        """
        Met à jour les données de marché avec le sentiment :
        - Calcule la moyenne pondérée du sentiment par symbole sur toutes les news du cycle.
        - Applique le score global sinon.
        - Enregistre tout dans shared_data.json pour usage persistant.
        """
        from collections import defaultdict

        # 1. Agrégation pondérée des scores par symbole
        symbol_sentiments = defaultdict(list)
        for item in sentiment_scores:
            symbols = item.get("symbols", [])
            score = item.get("sentiment", 0)
            if not symbols:
                # PATCH: Appliquer le score global à toutes les paires
                for key in self.market_data:
                    self.market_data[key]["sentiment"] = score
                    self.market_data[key]["sentiment_timestamp"] = time.time()
                continue
            for symbol in symbols:
                symbol = symbol.upper()
                for key in self.market_data:
                    if symbol in key.upper():
                        self.market_data[key]["sentiment"] = score
                        self.market_data[key]["sentiment_timestamp"] = time.time()
                        print(
                            f"[DEBUG SENTIMENT FUZZY ASSIGN] {key} <- {score} via symbol={symbol}"
                        )

        # 2. Applique la moyenne pondérée à chaque paire
        for key in self.market_data:
            # Extrait le ticker principal, ex: "BTCUSDT" -> "BTC", "ETHUSDT" -> "ETH"
            ticker = key.replace("USDT", "").replace("USD", "")
            values = symbol_sentiments.get(ticker, [])
            if values:
                total = sum(s * i for s, i in values)
                total_weight = sum(i for _, i in values)
                avg = total / total_weight if total_weight else 0
                self.market_data[key]["sentiment"] = avg
                self.market_data[key]["sentiment_timestamp"] = time.time()
                print(
                    f"[DEBUG AGG SENTIMENT] {key} <- {avg:.4f} via {len(values)} news (pondérée)"
                )

        # 3. Récupère la valeur globale du sentiment depuis le fichier partagé
        try:
            self.safe_update_shared_data(
                {
                    "last_sentiment_update": time.time(),
                    "sentiment_by_symbol": symbol_sentiments_out,
                },
                self.data_file,
            )
            news_sentiment = shared_data.get("sentiment", None)
            if news_sentiment:
                global_sentiment = news_sentiment.get("overall_sentiment", 0)
            else:
                global_sentiment = 0
        except Exception as e:
            print(f"[DEBUG ERROR] Could not read global sentiment from file: {e}")
            global_sentiment = 0

        print(f"[DEBUG SENTIMENT GLOBAL FINAL] avg_sentiment={global_sentiment}")

        # 4. Applique le score global si aucune news spécifique
        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            if pair_key not in self.market_data:
                self.market_data[pair_key] = {}

            if (
                "sentiment" not in self.market_data[pair_key]
                or self.market_data[pair_key]["sentiment"] == 0
            ):
                self.market_data[pair_key]["sentiment"] = global_sentiment
                self.market_data[pair_key]["sentiment_timestamp"] = time.time()
                print(
                    f"[DEBUG PROPAG GLOBAL SENTIMENT] {pair_key} <- {global_sentiment}"
                )

        # 5. Sauvegarde tous les sentiments dans shared_data.json
        symbol_sentiments_out = {
            key: data.get("sentiment", 0) for key, data in self.market_data.items()
        }
        try:
            self.safe_update_shared_data(
                {
                    "last_sentiment_update": time.time(),
                    "sentiment_by_symbol": symbol_sentiments_out,
                },
                self.data_file,
            )
            print("[SENTIMENT SAVE] shared_data.json mis à jour avec les sentiments")
        except Exception as e:
            print(f"[SENTIMENT SAVE ERROR] {e}")

    # Remplace la méthode async def _save_sentiment_data(...) par la version patchée ci-dessous :
    async def _save_sentiment_data(self, sentiment_scores, news_data=None):
        """
        Enregistre les données de sentiment du marché (scores, news, global) dans le fichier partagé.
        Correction : merge les news pour préserver le champ 'processed' à chaque sauvegarde.
        """
        headlines = []
        if news_data is None:
            news_data = sentiment_scores
        if isinstance(news_data, list):
            for item in news_data[:10]:
                if isinstance(item, dict) and "title" in item:
                    headlines.append(
                        str(item["title"])
                    )  # Toujours str pour éviter erreur

        # Correction : on prend les scores assignés dans market_data
        valid_scores = [
            data.get("sentiment")
            for key, data in self.market_data.items()
            if data.get("sentiment") is not None
        ]
        print(
            f"[DEBUG _save_sentiment_data] valid_scores from market_data={valid_scores}"
        )

        # Fallback sur sentiment_scores si jamais
        if not valid_scores:
            valid_scores = [
                item.get("sentiment")
                for item in sentiment_scores
                if isinstance(item, dict) and item.get("sentiment") is not None
            ]
            print(
                f"[DEBUG _save_sentiment_data] fallback valid_scores from sentiment_scores={valid_scores}"
            )

        # === PATCH : Utilise le nouveau résumé pour remplir le sentiment_data ===
        summary = get_sentiment_summary_from_batch(sentiment_scores)
        sentiment_global = summary["sentiment_global"]
        impact_score = float(
            np.mean(
                [
                    abs(item.get("sentiment", 0))
                    for item in sentiment_scores
                    if isinstance(item, dict)
                ]
            )
            if sentiment_scores
            else 0.0
        )
        major_events = (
            "; ".join(summary["top_news"][:3]) if summary["top_news"] else "Aucun"
        )

        print(
            f"[DEBUG SENTIMENT GLOBAL] sentiment_global={sentiment_global} impact={impact_score} major_events={major_events}"
        )

        sentiment_data = {
            "timestamp": datetime.now().isoformat(),
            "scores": sentiment_scores,
            "latest_news": summary["top_news"],
            "overall_sentiment": sentiment_global,
            "impact_score": impact_score,
            "major_events": major_events,
            "top_symbols": summary["top_symbols"],
            "n_news": summary["n_news"],
        }

        # === PATCH : Merge des news pour préserver "processed" ===
        try:
            with open(self.data_file, "r") as f:
                shared_data_prev = json.load(f)
            old_scores = shared_data_prev.get("sentiment", {}).get("scores", [])
        except Exception:
            old_scores = []

        sentiment_data["scores"] = merge_news_processed(
            old_scores, sentiment_data["scores"]
        )

        try:
            self.safe_update_shared_data({"sentiment": sentiment_data}, self.data_file)
            self.logger.info(
                f"[SENTIMENT] Data written successfully to {self.data_file}"
            )
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")

    async def generate_market_analysis_report(self, cycle=None):
        debug_market_data_structure(
            self.market_data, self.pairs_valid, ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
        report = (
            f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {get_current_time()}\n"
            f"Cycle: {cycle if cycle is not None else self.current_cycle}\n"
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

        # --- AJOUT : Section news/sentiment globale détaillée ---
        try:
            with open(self.data_file, "r") as f:
                shared_data = json.load(f)
            news_sentiment = shared_data.get("sentiment", None)
        except Exception:
            news_sentiment = None

        if news_sentiment and isinstance(news_sentiment, dict):
            try:
                sentiment = float(news_sentiment.get("overall_sentiment", 0) or 0)
            except Exception:
                sentiment = 0.0
            try:
                impact = float(news_sentiment.get("impact_score", 0) or 0)
            except Exception:
                impact = 0.0
            major_events = news_sentiment.get("major_events", "Aucun")
            report += (
                "\n📰 Analyse des News:\n"
                f"Sentiment: {sentiment:.2%}\n"
                f"Impact estimé: {impact:.2%}\n"
                f"Événements majeurs: {major_events}\n"
            )
            # Ajout des dernières news si dispo
            major_news = news_sentiment.get("latest_news", [])
            if major_news:
                report += "Dernières news :\n"
                for news in major_news[:3]:
                    report += f"- {news}\n"
        else:
            report += "\n📰 Analyse des News: Aucune donnée disponible.\n"

        # Ajout des informations de sentiment par paire si disponibles
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
            closes = [
                c for c in data.get("close", []) if c is not None and not np.isnan(c)
            ]
            if len(closes) < 10:
                return 0.0
            closes = closes[-20:]
            if len(closes) < 10:
                return 0.0
            ma_fast = np.mean(closes[-10:])
            ma_slow = np.mean(closes)
            if ma_slow == 0 or np.isnan(ma_fast) or np.isnan(ma_slow):
                return 0.0
            trend = (ma_fast / ma_slow) - 1
            return float(trend)
        except Exception as e:
            print("DEBUG calculate_trend error:", e)
            return 0.0

    def calculate_volatility(self, data):
        try:
            closes = [
                c
                for c in data.get("close", [])
                if c is not None and not np.isnan(c) and c > 0
            ]
            if len(closes) < 2:
                return 0.0
            closes = closes[-20:]
            if len(closes) < 2 or any(c <= 0 for c in closes):
                return 0.0  # protège contre log(0) ou log négatif
            returns = np.diff(np.log(closes))
            if np.isnan(returns).any() or np.isinf(returns).any():
                return 0.0
            return float(np.std(returns) * np.sqrt(252))
        except Exception as e:
            print("DEBUG calculate_volatility error:", e)
            return 0.0

    def calculate_volume_profile(self, data):
        try:
            if isinstance(data, dict) and "volume" in data:
                volumes = data["volume"][-20:]
                if not volumes or len(volumes) < 2:
                    return {"strength": 1.0}
                current_vol = volumes[-1]
                avg_vol = sum(volumes) / len(volumes)
                return {
                    "strength": float(current_vol / avg_vol) if avg_vol > 0 else 1.0
                }
            return {"strength": 1.0}
        except Exception as e:
            print("DEBUG calculate_volume_profile error:", e)
            return {"strength": 1.0}

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
                    vol_dict = self.calculate_volume_profile(data)
                    # Sécurisation : toujours prendre la clé 'strength' si c'est un dict
                    if isinstance(vol_dict, dict):
                        vol = vol_dict.get("strength", 1.0)
                    else:
                        vol = vol_dict  # fallback : float direct si jamais
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
                self.market_data.get("BTCUSDC", {}).get("1h", {})
            )
            trend = self.calculate_trend(
                self.market_data.get("BTCUSDC", {}).get("1h", {})
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

            log_dashboard(
                f"[MARKET ANALYSIS] Régime détecté: {self.regime} | Volatilité: {volatility:.4f} | Tendance: {trend:.4f}"
            )

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
        """
        Ajoute les prédictions des modèles d'IA aux données de marché.
        Corrige dynamiquement le shape de ppo_features selon le nombre de paires.
        """
        # PATCH: Définit les constantes locales nécessaires !
        N_STEPS = self.N_STEPS
        N_FEATURES = self.N_FEATURES

        if not self.ai_enabled or not self.dl_model or not self.ppo_strategy:
            return

        expected_shape = (self.get_input_dim(),)
        num_pairs = len(self.pairs_valid)

        ppo_features_list = []
        dl_predictions = {}

        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            features = await self._prepare_features_for_ai(pair_key)
            if features is not None:
                try:
                    # Prédiction du CNN-LSTM
                    dl_prediction = self.dl_model.predict(features)
                    dl_predictions[pair_key] = dl_prediction

                    # Correction NaN/inf
                    for k in features:
                        arr = features[k]
                        if isinstance(arr, np.ndarray):
                            if np.isnan(arr).any() or np.isinf(arr).any():
                                print(
                                    f"[WARN] NaN/inf détecté dans {k}, correction appliquée"
                                )
                                features[k] = np.nan_to_num(arr)
                        else:
                            if np.isnan(features[k]) or np.isinf(features[k]):
                                print(
                                    f"[WARN] NaN/inf détecté dans {k}, correction appliquée"
                                )
                                features[k] = float(np.nan_to_num(features[k]))

                    # Construction du vecteur feature
                    vec = np.concatenate(
                        [
                            (
                                features[k]
                                if isinstance(features[k], np.ndarray)
                                else np.full(N_STEPS, features[k])
                            )
                            for k in [
                                "close",
                                "high",
                                "low",
                                "volume",
                                "rsi",
                                "macd",
                                "volatility",
                                "vol_ratio",
                            ]
                        ]
                    )
                    if vec.shape != (N_FEATURES * N_STEPS,):
                        print(
                            f"[SKIP PPO] {pair_key}, shape {vec.shape}, pas assez de data"
                        )
                        continue
                    ppo_features_list.append(vec)
                except Exception as e:
                    self.logger.error(f"Error preparing AI features for {pair}: {e}")

        if not ppo_features_list:
            print("[SKIP PPO] Aucun vecteur de features disponible pour PPO.")
            return
        ppo_features = np.concatenate(ppo_features_list)
        expected_shape = (N_FEATURES * N_STEPS * num_pairs,)
        print(
            f"[DEBUG] Shape du vecteur features PPO : {ppo_features.shape}, attendu : {expected_shape}"
        )
        if ppo_features.shape != expected_shape:
            print(f"[SKIP PPO] Shape {ppo_features.shape}, attendu: {expected_shape}")
            return

        print("PPO features shape:", ppo_features.shape)

        try:
            ppo_action = self.ppo_strategy.get_action(ppo_features)
            for i, pair in enumerate(self.pairs_valid):
                pair_key = pair.replace("/", "").upper()
                dl_pred = dl_predictions.get(pair_key, 0)
                await self._merge_signals(pair_key, dl_pred, ppo_action)
        except Exception as e:
            self.logger.error(f"Error getting PPO action: {e}")

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

    def initialize_shared_data(self):
        """
        Initialise le fichier partagé SANS écraser l'historique :
        - Conserve l'existant (news, positions, historiques…)
        - Réinitialise uniquement certains champs (cycle, régime, performance…)
        """
        # Charge l'existant si présent
        if os.path.exists(self.data_file):
            with open(self.data_file, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Réinitialise uniquement les champs nécessaires
        data["timestamp"] = get_current_time()
        data["user"] = CURRENT_USER

        # PATCH : Réinit de bot_status
        data["bot_status"] = {
            "regime": self.regime,
            "cycle": 0,  # Cycle remis à zéro !
            "last_update": get_current_time(),
            "performance": {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "balance": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0,
                "total_loss": 0,
            },
        }

        # Optionnel : tu peux ajouter ici d'autres champs à réinitialiser si besoin
        # Exemple :
        # data["active_pauses"] = []
        # data["equity_history"] = []

        # NE PAS TOUCHER aux autres champs : news, positions, historiques, etc.

        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=4)

    def save_shared_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            # MAJ des sections
            data.update(
                {
                    "timestamp": get_current_time(),
                    "user": CURRENT_USER,
                    "bot_status": {
                        "regime": self.regime,
                        "cycle": self.current_cycle,
                        "last_update": get_current_time(),
                        "performance": self.get_performance_metrics(),
                    },
                    "market_data": self.market_data,
                    "indicators": self.indicators,
                }
            )

            # Ajoute les métriques avancées pour dashboard
            perf = data["bot_status"]["performance"]
            equity_history = data.get("equity_history", [])
            if equity_history and len(equity_history) > 10:
                import numpy as np

                balances = [pt["balance"] for pt in equity_history if "balance" in pt]
                perf["max_drawdown"] = float(
                    np.min(
                        [0]
                        + [
                            (min(balances[i:], default=0) - b) / b
                            for i, b in enumerate(balances)
                            if b > 0
                        ]
                    )
                )
                returns = np.diff(np.array(balances)) / np.array(balances)[:-1]
                perf["sharpe_ratio"] = (
                    float(np.mean(returns) / np.std(returns))
                    if np.std(returns) > 0
                    else 0
                )
            data["bot_status"]["performance"] = perf

            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving shared data: {e}")

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
                "balance": 0 + (self.current_cycle * 100),
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
        Calcule tous les indicateurs nécessaires pour les stratégies du dossier 'strategies'.
        Retourne un dictionnaire {nom_indicateur: dernière_valeur non-NaN ou None}
        (Version enrichie avec indicateurs avancés)
        Corrige définitivement le warning VWAP/VWMA not datetime ordered de pandas-ta !
        """
        import pandas as pd
        import numpy as np

        try:
            # --- Conversion stricte et tri ---
            # Si df est une liste, transforme-le en DataFrame
            if isinstance(df, list):
                if len(df) == 0:
                    self.logger.error("add_indicators: Liste reçue vide")
                    return None
                if isinstance(df[0], dict):
                    df = pd.DataFrame(df)
                elif isinstance(df[0], (list, tuple)):
                    columns = ["timestamp", "open", "high", "low", "close", "volume"]
                    df = pd.DataFrame(df, columns=columns)
                else:
                    self.logger.error(
                        "add_indicators: Format de liste non pris en charge"
                    )
                    return None

            if not isinstance(df, pd.DataFrame):
                self.logger.error("add_indicators: df n'est pas un DataFrame")
                return None

            # --- Vérification et correction colonne timestamp ---
            if "timestamp" not in df.columns:
                self.logger.error("add_indicators: colonne 'timestamp' manquante")
                return None

            # --- Conversion stricte timestamp ---
            try:
                # Si timestamp n'est pas datetime, convertis-le
                if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
                    # Si c'est en ms, convertis-le
                    # Heuristique: timestamp > 1e12 => probablement en ms
                    if df["timestamp"].max() > 1e12:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    else:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception as e:
                self.logger.error(f"add_indicators: Erreur conversion timestamp: {e}")
                return None

            # --- Tri strict ---
            df = df.drop_duplicates(subset="timestamp", keep="last")
            df = df.sort_values("timestamp")
            df = df.reset_index(drop=True)

            required_cols = {"open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                self.logger.error(
                    f"add_indicators: Colonnes manquantes: {required_cols - set(df.columns)} | Colonnes actuelles: {df.columns.tolist()}"
                )
                return None

            MIN_LEN = 30
            if df is None or len(df) < MIN_LEN:
                self.logger.warning(
                    f"DataFrame vide ou insuffisant ({0 if df is None else len(df)}) lignes"
                )
                return None

            if df.empty:
                self.logger.warning(
                    "DataFrame vide, impossible de calculer les indicateurs"
                )
                print("[DEBUG add_indicators] DataFrame vide après tri/formatage")
                return None

            try:
                df_ta = df.copy()

                # Tri STRICT + SET INDEX avant CHAQUE calcul d'indicateur avancé (VWMA, VWAP, OBV, etc.)
                def strict_sort_and_index(df):
                    if "timestamp" in df.columns:
                        df = df.drop_duplicates(subset="timestamp", keep="last")
                        df = df.sort_values("timestamp")
                        df = df.set_index("timestamp")
                    return df

                # Calcul des indicateurs classiques (index classique)
                sma_20 = df_ta.ta.sma(length=20, append=False)
                if sma_20 is not None and not sma_20.empty:
                    if isinstance(sma_20, pd.Series):
                        df_ta["sma_20"] = sma_20
                    elif "SMA_20" in sma_20:
                        df_ta["sma_20"] = sma_20["SMA_20"]

                sma_50 = df_ta.ta.sma(length=50, append=False)
                if sma_50 is not None and not sma_50.empty:
                    if isinstance(sma_50, pd.Series):
                        df_ta["sma_50"] = sma_50
                    elif "SMA_50" in sma_50:
                        df_ta["sma_50"] = sma_50["SMA_50"]

                ema_20 = df_ta.ta.ema(length=20, append=False)
                if ema_20 is not None and not ema_20.empty:
                    if isinstance(ema_20, pd.Series):
                        df_ta["ema_20"] = ema_20
                    elif "EMA_20" in ema_20:
                        df_ta["ema_20"] = ema_20["EMA_20"]

                rsi_14 = df_ta.ta.rsi(length=14, append=False)
                if rsi_14 is not None and not rsi_14.empty:
                    if isinstance(rsi_14, pd.Series):
                        df_ta["rsi_14"] = rsi_14
                    elif "RSI_14" in rsi_14:
                        df_ta["rsi_14"] = rsi_14["RSI_14"]

                macd = df_ta.ta.macd()
                if macd is not None and not macd.empty:
                    if "MACD_12_26_9" in macd:
                        df_ta["macd"] = macd["MACD_12_26_9"]
                    if "MACDs_12_26_9" in macd:
                        df_ta["macd_signal"] = macd["MACDs_12_26_9"]
                    if "MACDh_12_26_9" in macd:
                        df_ta["macd_hist"] = macd["MACDh_12_26_9"]

                bb = df_ta.ta.bbands(length=20, std=2.0)
                if bb is not None and not bb.empty:
                    if "BBL_20_2.0" in bb:
                        df_ta["bb_lower"] = bb["BBL_20_2.0"]
                    if "BBU_20_2.0" in bb:
                        df_ta["bb_upper"] = bb["BBU_20_2.0"]

                df_ta["donchian_high"] = df_ta["high"].rolling(window=20).max()
                df_ta["donchian_low"] = df_ta["low"].rolling(window=20).min()

                psar = df_ta.ta.psar()
                if psar is not None and not psar.empty:
                    key = [col for col in psar.columns if col.startswith("PSAR")][0]
                    df_ta["psar"] = psar[key]

                mom_10 = df_ta.ta.mom(length=10, append=False)
                if mom_10 is not None and not mom_10.empty:
                    if isinstance(mom_10, pd.Series):
                        df_ta["momentum_10"] = mom_10
                    elif "MOM_10" in mom_10:
                        df_ta["momentum_10"] = mom_10["MOM_10"]

                df_ta["zscore_20"] = (
                    df_ta["close"] - df_ta["close"].rolling(20).mean()
                ) / df_ta["close"].rolling(20).std()

                # Indicateurs avancés supplémentaires : TRI + SET INDEX obligatoire pour pandas-ta VWAP/VWMA/OBV
                try:
                    df_ta_idx = strict_sort_and_index(df_ta)
                    vwma = df_ta_idx.ta.vwma(length=20)
                    # On remet l'index timestamp dans la colonne pour rester compatible
                    df_ta["vwma_20"] = vwma.values
                except Exception:
                    df_ta["vwma_20"] = np.nan
                try:
                    df_ta_idx = strict_sort_and_index(df_ta)
                    obv = df_ta_idx.ta.obv()
                    df_ta["obv"] = obv.values
                except Exception:
                    df_ta["obv"] = np.nan
                try:
                    df_ta_idx = strict_sort_and_index(df_ta)
                    vwap = df_ta_idx.ta.vwap()
                    df_ta["vwap"] = vwap.values
                except Exception:
                    df_ta["vwap"] = np.nan
                try:
                    stochrsi = df_ta.ta.stochrsi()
                    if stochrsi is not None and not stochrsi.empty:
                        df_ta["stochrsi"] = stochrsi.iloc[:, 0]
                except Exception:
                    df_ta["stochrsi"] = np.nan
                try:
                    kc = df_ta.ta.kc()
                    if kc is not None and not kc.empty:
                        df_ta["kc_upper"] = kc["KCUpper_20_2_10"]
                        df_ta["kc_lower"] = kc["KCLower_20_2_10"]
                except Exception:
                    df_ta["kc_upper"] = df_ta["kc_lower"] = np.nan
                try:
                    supertrend = df_ta.ta.supertrend(length=7, multiplier=3.0)
                    if supertrend is not None and not supertrend.empty:
                        df_ta["supertrend"] = supertrend.iloc[:, 0]
                except Exception:
                    pass
                try:
                    ichimoku = df_ta.ta.ichimoku()
                    if ichimoku is not None and not ichimoku.empty:
                        df_ta["ichimoku_a"] = ichimoku["ISA_9"]
                        df_ta["ichimoku_b"] = ichimoku["ISB_26"]
                except Exception:
                    pass
                try:
                    keltner = df_ta.ta.kc()
                    if keltner is not None and not keltner.empty:
                        df_ta["keltner_upper"] = keltner["KCUpper_20_2_10"]
                        df_ta["keltner_lower"] = keltner["KCLower_20_2_10"]
                except Exception:
                    pass
                try:
                    accdist = df_ta.ta.accdist()
                    df_ta["accumulation"] = accdist
                except Exception:
                    pass

                all_indics = [
                    "sma_20",
                    "sma_50",
                    "ema_20",
                    "rsi_14",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "bb_lower",
                    "bb_upper",
                    "donchian_high",
                    "donchian_low",
                    "psar",
                    "momentum_10",
                    "zscore_20",
                    "vwma_20",
                    "obv",
                    "vwap",
                    "stochrsi",
                    "kc_upper",
                    "kc_lower",
                ]

                indicators = {}
                for col in all_indics:
                    if col in df_ta.columns:
                        last_valid = df_ta[col].dropna()
                        indicators[col] = (
                            last_valid.iloc[-1] if not last_valid.empty else None
                        )
                    else:
                        indicators[col] = None

            except Exception as e:
                self.logger.warning(f"Erreur pandas-ta indicateurs principaux : {e}")
                indicators = {}

            n_valid = len([v for v in indicators.values() if v is not None])
            self.logger.info(
                f"✅ {n_valid} indicateurs extraits automatiquement sur {df.shape[0]} lignes"
            )
            print(
                f"[DEBUG add_indicators] {n_valid} indicateurs extraits: {list(indicators.keys())[:5]}"
            )
            return indicators

        except Exception as e:
            self.logger.error(f"❌ Erreur calcul indicateurs: {e}")
            return None

    def train_cnn_lstm_on_live(self, pair="BTCUSDT", tf="1h"):
        """
        Entraîne le modèle CNN-LSTM sur les données live de ws_collector pour la paire/timeframe donnée,
        et sauvegarde les poids dans src/models/cnn_lstm_model.pth
        (NE RESET PLUS à cause de NaN/inf)
        """
        try:
            from src.ai.train_cnn_lstm import train_with_live_data
        except ImportError:
            print("Impossible d'importer train_with_live_data")
            return
        pair_key = pair.replace("/", "").upper()
        print(f"Chargement du DataFrame live pour {pair_key} / {tf}")
        print(
            f"[DEBUG] ws_collector.get_dataframe({pair_key}, {tf}) keys: {list(self.ws_collector.data.keys()) if hasattr(self.ws_collector, 'data') else 'no data attr'}"
        )
        df_live = self.ws_collector.get_dataframe(pair_key, tf)
        if df_live is not None and not df_live.empty:
            df_live = add_dl_features(df_live)
            # Ici : plus jamais de reset si NaN/inf, on log juste le nombre de NaN restant
            for col in ["rsi", "macd", "volatility"]:
                n_nan = df_live[col].isna().sum() if col in df_live.columns else 0
                if n_nan > 0:
                    print(f"⚠️ Attention : {n_nan} NaN dans {col} même après correction")
            print(f"Entraînement du modèle IA sur {len(df_live)} lignes live…")
            train_with_live_data(df_live)
        else:
            print("Aucune donnée live disponible pour entraîner le modèle.")

    def train_cnn_lstm_on_all_live(self):
        """
        Entraîne le modèle CNN-LSTM sur toutes les paires et timeframes de la config,
        en utilisant les données live du ws_collector.
        (NE RESET PLUS à cause de NaN/inf)
        """
        try:
            from src.ai.train_cnn_lstm import train_with_live_data
        except ImportError:
            print("Impossible d'importer train_with_live_data")
            return

        for pair in self.pairs_valid:
            pair_key = pair.replace("/", "").upper()
            for tf in self.config["TRADING"]["timeframes"]:
                print(f"→ Entraînement IA sur {pair_key} / {tf}")
                print(
                    f"[DEBUG] ws_collector.get_dataframe({pair_key}, {tf}) keys: {list(self.ws_collector.data.keys()) if hasattr(self.ws_collector, 'data') else 'no data attr'}"
                )
                df_live = self.ws_collector.get_dataframe(pair_key, tf)
                if df_live is not None and not df_live.empty:
                    df_live = add_dl_features(df_live)
                    for col in ["rsi", "macd", "volatility"]:
                        n_nan = (
                            df_live[col].isna().sum() if col in df_live.columns else 0
                        )
                        if n_nan > 0:
                            print(
                                f"⚠️ Attention : {n_nan} NaN dans {col} même après correction"
                            )
                    print(
                        f"  {len(df_live)} lignes live trouvées, entraînement en cours…"
                    )
                    train_with_live_data(df_live)
                else:
                    print(f"  Pas de données live pour {pair_key} / {tf}, skip.")


def filter_pairs(
    bot,
    min_volatility=0.01,
    min_signal=0.3,
    top_n=5,
    vol_anomaly_filter=True,
    vol_threshold=0.12,
    anomaly_threshold=4.0,
):
    """
    Filtre dynamiquement les paires selon :
    - Volatilité
    - Score du signal
    - Propreté du marché
    - NOUVEAU: Corrélations entre paires
    """
    from src.analysis.filters.volatility_anomaly_filter import filter_market
    from src.analysis.filters.correlation_filter import filter_uncorrelated_pairs
    import numpy as np
    import pandas as pd

    # NOUVEAU: Calcul des corrélations
    correlations = bot.calculate_correlation_matrix()

    candidates = []
    for pair in bot.pairs_valid:
        pair_key = pair.replace("/", "").upper()

        # 1. Analyse volatilité
        vol = 0
        if (
            pair_key in bot.market_data
            and "1h" in bot.market_data[pair_key]
            and "close" in bot.market_data[pair_key]["1h"]
        ):

            closes = bot.market_data[pair_key]["1h"]["close"]
            if len(closes) >= 20:
                returns = np.diff(np.log(closes[-20:]))
                vol = float(np.std(returns))

        # 2. Récupération signal
        signal = 0
        if pair_key in bot.market_data and "ai_prediction" in bot.market_data[pair_key]:
            signal = bot.market_data[pair_key]["ai_prediction"]

        # 3. Analyse anomalies
        df_ohlcv = None
        if (
            pair_key in bot.market_data
            and "1h" in bot.market_data[pair_key]
            and all(
                k in bot.market_data[pair_key]["1h"]
                for k in ["close", "high", "low", "volume"]
            )
        ):

            df_ohlcv = pd.DataFrame(
                {
                    "close": bot.market_data[pair_key]["1h"]["close"],
                    "high": bot.market_data[pair_key]["1h"]["high"],
                    "low": bot.market_data[pair_key]["1h"]["low"],
                    "volume": bot.market_data[pair_key]["1h"]["volume"],
                }
            )

        # 4. Vérification propreté marché
        is_clean = True
        if vol_anomaly_filter and df_ohlcv is not None and len(df_ohlcv) >= 50:
            is_clean = filter_market(
                df_ohlcv,
                vol_threshold=vol_threshold,
                anomaly_threshold=anomaly_threshold,
                price_col="close",
            )

        # 5. NOUVEAU: Score de corrélation
        corr_score = max([v for k, v in correlations.items() if pair in k], default=1.0)

        # 6. Score final combiné
        final_score = (vol * abs(signal)) * (1 - corr_score)

        print(f"[FILTER DEBUG] {pair_key}:")
        print(f"  - Volatilité: {vol:.4f}")
        print(f"  - Signal: {signal:.4f}")
        print(f"  - Corrélation: {corr_score:.4f}")
        print(f"  - Marché propre: {is_clean}")
        print(f"  - Score final: {final_score:.4f}")

        if is_clean and final_score > min_signal:
            candidates.append((pair, final_score))
            print(f"✅ {pair_key} ACCEPTÉ")
        else:
            print(f"❌ {pair_key} REJETÉ")

    # Tri par score final
    candidates.sort(key=lambda x: x[1], reverse=True)
    filtered_candidates = [c[0] for c in candidates]

    # Filtrage corrélation final
    filtered_uncorr = filter_uncorrelated_pairs(
        bot.market_data,
        filtered_candidates,
        timeframe="1h",
        window=50,
        corr_threshold=0.85,
        top_n=top_n,
    )

    return filtered_uncorr


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
        bot = None
        try:
            print("\n=== DÉMARRAGE DU BOT ===")
            print("🚀 Trading Bot Ultimate v4 - Version Ultra-Propre")

            # 1. Configuration initiale
            valid_pairs = load_config()

            # 2. Création et configuration du bot
            bot = TradingBotM4()
            bot.pairs_valid = valid_pairs

            # 3. Préchargement historique (optionnel, sécurisé)
            if hasattr(bot, "ws_collector") and hasattr(bot, "binance_client"):
                for symbol in bot.pairs_valid:  # PAS bot.config["TRADING"]["pairs"]
                    symbol_binance = symbol.replace("/", "").upper()
                    for tf in bot.config["TRADING"]["timeframes"]:
                        try:
                            bot.ws_collector.preload_historical(
                                bot.binance_client, symbol_binance, tf, limit=2000
                            )
                            print(f"Préchargement {symbol_binance} {tf} OK")
                        except Exception as e:
                            print(f"Erreur préchargement {symbol_binance} {tf} : {e}")

            # === AJOUT DIAGNOSTIC DATAFRAME ===
            if hasattr(bot, "ws_collector"):
                print("\n=== DIAGNOSTIC : Contenu du ws_collector ===")
                for pair in bot.pairs_valid:
                    pair_key = pair.replace("/", "").upper()
                    for tf in bot.config["TRADING"]["timeframes"]:
                        df = bot.ws_collector.get_dataframe(pair_key, tf)
                        print(
                            f"{pair_key}-{tf}: {len(df) if df is not None else 0} lignes"
                        )
                print("=== FIN DIAGNOSTIC ===\n")
            # === FIN AJOUT ===

            # 4. Setup des composants internes (websockets, news, etc)
            ok = await bot._setup_components()
            if not ok:
                print("❌ Echec de l'initialisation des composants.")
                return None, None

            # 5. Chargement des données de marché réelles si trading live
            if getattr(bot, "is_live_trading", False):
                await bot._fetch_real_market_data()
                for sym in bot.market_data:
                    print(f"{sym}: {list(bot.market_data[sym].keys())}")

            # 6. Premier rapport d'analyse
            try:
                initial_report = await bot.generate_market_analysis_report(cycle=0)
            except Exception as e:
                initial_report = (
                    f"[ERREUR] Impossible de générer le rapport initial: {e}"
                )

            # 7. Envoi du message Telegram d'initialisation
            try:
                await bot.telegram.send_message(
                    "🚀 <b>Bot Trading démarré</b>\n"
                    "✅ Initialisation réussie\n"
                    f"📊 Paires configurées: {', '.join(valid_pairs)}\n\n"
                    f"{initial_report}"
                )
            except Exception as e:
                print(f"Erreur lors de l'envoi Telegram : {e}")

            print("✅ Bot initialized successfully")
            return bot, valid_pairs

        except Exception as e:
            logger.error(f"Erreur d'initialisation: {e}", exc_info=True)
            print(f"❌ ERREUR FATALE lors de l'initialisation: {e}")
            return None, None

    async def market_analysis_cycle(bot, pair, market_data, tf="1h"):
        try:
            pair_key = pair.replace("/", "").upper()
            if not market_data or pair_key not in market_data:
                return None

            ohlcv_df = bot.ws_collector.get_dataframe(pair_key, tf)
            if ohlcv_df is None or len(ohlcv_df) < 20:
                return None

            indicators_data = bot.add_indicators(ohlcv_df)

            # === PATCH AUTO-STRATEGIE ===
            if hasattr(bot, "auto_strategy_config") and bot.auto_strategy_config:
                auto_cfg = bot.auto_strategy_config
                if (
                    pair_key.upper() == auto_cfg["pair"].upper()
                    and tf == auto_cfg["timeframe"]
                ):
                    action = appliquer_config_strategy(ohlcv_df, auto_cfg["config"])
                    signal = {"action": action, "confidence": 1.0}
                else:
                    # Appel standard
                    signal = await bot.analyze_signals(
                        pair_key, ohlcv_df, indicators_data, tf=tf
                    )
                    signal["pair"] = pair
                    signal["tf"] = tf
                    return signal
            else:
                # Appel standard
                signal = await bot.analyze_signals(
                    pair_key, ohlcv_df, indicators_data, tf=tf
                )
                signal["pair"] = pair
                signal["tf"] = tf
                return signal
            # === FIN PATCH AUTO-STRATEGIE ===

            return signal

        except Exception as e:
            logger.error(f"Erreur analyse {pair}: {e}")
            return None

    async def execute_trading_cycle(bot, valid_pairs):
        """Exécute un cycle complet de trading (fusion multi-timeframe optimisée)"""
        try:
            # 0. Import avancé des indicateurs orderflow
            try:
                from src.analysis.technical.advanced.advanced_indicators import (
                    AdvancedIndicators,
                )

                orderflow_indicators = AdvancedIndicators()
            except Exception as e:
                orderflow_indicators = None
                print("[Orderflow] Impossible d'importer AdvancedIndicators:", e)

            # 1. Injection des données live WS dans market_data
            for pair in bot.pairs_valid:
                pair_key = pair.replace("/", "").upper()
                if pair_key not in bot.market_data:
                    bot.market_data[pair_key] = {}
                for tf in bot.config["TRADING"]["timeframes"]:
                    df = bot.ws_collector.get_dataframe(pair_key, tf)
                    # DEBUG: Ajout log sur la taille du DataFrame
                    print(
                        f"[DEBUG] DataFrame {pair_key}-{tf}: {len(df) if df is not None else 'None'} lignes"
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
                        # 1bis. Indicateurs orderflow
                        if orderflow_indicators is not None:
                            try:
                                bid_ask = (
                                    orderflow_indicators._bid_ask_ratio(df)
                                    if hasattr(orderflow_indicators, "_bid_ask_ratio")
                                    else None
                                )
                                liquidity_wave = (
                                    orderflow_indicators._liquidity_wave(df)
                                    if hasattr(orderflow_indicators, "_liquidity_wave")
                                    else None
                                )
                                smart_money = (
                                    orderflow_indicators._smart_money_index(df)
                                    if hasattr(
                                        orderflow_indicators, "_smart_money_index"
                                    )
                                    else None
                                )
                                bot.market_data[pair_key][tf]["orderflow"] = {
                                    "bid_ask_ratio": bid_ask,
                                    "liquidity_wave": liquidity_wave,
                                    "smart_money_index": smart_money,
                                }
                            except Exception as e:
                                print(f"[Orderflow] Erreur calcul {pair_key} {tf}: {e}")
                    else:
                        print(f"[DEBUG] DataFrame vide pour {pair_key} {tf}")

            # 2. Analyse de marché
            regime, market_data, indicators = await bot.study_market("7d")
            strategy = bot.choose_strategy(regime, indicators)
            log_dashboard(f"🎯 Stratégie active: {strategy}")

            # 3. Détection d'arbitrage
            await handle_arbitrage_opportunities(bot)

            # 4. Analyse des paires pour CHAQUE timeframe (génère signaux bruts multi-tf)
            SHARED_DATA_PATH = "src/shared_data.json"
            try:
                with open(SHARED_DATA_PATH, "r") as f:
                    shared_data = json.load(f)
                f_params = shared_data.get("filtering_params", {})
                min_vol = float(f_params.get("min_volatility", 0.01))
                min_sig = float(f_params.get("min_signal", 0.3))
                n_top = int(f_params.get("top_n", 5))
            except Exception:
                min_vol, min_sig, n_top = 0.01, 0.3, 5

            selected_pairs = bot.pairs_valid
            # selected_pairs = filter_pairs(
            # bot, min_volatility=0.0, min_signal=0.0, top_n=10
            # )
            print(selected_pairs)
            print(f"[DYNAMIQUE] Paires sélectionnées ce cycle : {selected_pairs}")
            ignored_pairs = [p for p in bot.pairs_valid if p not in selected_pairs]
            if ignored_pairs:
                print(
                    f"[DYNAMIQUE] Paires ignorées (pas d'opportunité) : {ignored_pairs}"
                )

            trade_decisions = []
            for pair in selected_pairs:
                for tf in bot.config["TRADING"]["timeframes"]:
                    df = bot.ws_collector.get_dataframe(
                        pair.replace("/", "").upper(), tf
                    )
                    print(
                        f"[DEBUG] Analyse {pair}-{tf} : {len(df) if df is not None else 'None'} lignes"
                    )
                    indicators_data = (
                        bot.add_indicators(df)
                        if df is not None and not df.empty
                        else {}
                    )
                    print(f"[DEBUG] Indicateurs {pair}-{tf} : {indicators_data}")
                    decision = await market_analysis_cycle(
                        bot, pair, bot.market_data, tf=tf
                    )
                    print(f"[DEBUG] Signal {pair}-{tf} : {decision}")
                    if decision:
                        decision["tf"] = tf
                        trade_decisions.append(decision)

            # LOG toutes les décisions par timeframe
            for decision in trade_decisions:
                signals = decision.get("signals", {})
                log_dashboard(
                    f"[TRADE DECISION] {decision['pair']} | TF: {decision.get('tf','?')} | "
                    f"Action: {decision['action'].upper()} | Confiance: {decision['confidence']:.2f} | "
                    f"Tech: {signals.get('technical', 0):.2f} | "
                    f"IA: {signals.get('ai', 0):.2f} | "
                    f"Sentiment: {signals.get('sentiment', 0):.2f}"
                )

            # 5. FUSION multi-timeframe : 1 décision centrale par paire (pondérée)
            signals_by_pair = {pair: {} for pair in valid_pairs}
            for decision in trade_decisions:
                pair = decision["pair"]
                tf = decision.get("tf")
                if pair in signals_by_pair and tf:
                    signals_by_pair[pair][tf] = decision

            final_trade_decisions = []
            for pair, tf_signals in signals_by_pair.items():
                action, confidence = bot.aggregate_timeframe_signals(pair, tf_signals)
                all_details = [
                    (tf, d["action"], d["confidence"]) for tf, d in tf_signals.items()
                ]
                log_dashboard(
                    f"[FUSION] {pair}: {all_details} => FINAL: {action.upper()} ({confidence:.2f})"
                )
                signals_example = next(iter(tf_signals.values()), {})
                final_trade_decisions.append(
                    {
                        "pair": pair,
                        "action": action,
                        "confidence": confidence,
                        "signals": {
                            "details": tf_signals,
                            "technical": signals_example.get("signals", {}).get(
                                "technical", 0
                            ),
                            "ai": signals_example.get("signals", {}).get("ai", 0),
                            "sentiment": signals_example.get("signals", {}).get(
                                "sentiment", 0
                            ),
                        },
                    }
                )

            # 6. Exécution des trades FUSIONNÉS (1 seul trade/pair/cycle)
            await execute_trade_decisions(bot, final_trade_decisions)

            return final_trade_decisions, regime

        except Exception as e:
            logger.error(f"Erreur cycle trading: {e}")
            raise

    async def main():
        try:
            # Initialisation
            bot, valid_pairs = await initialize_bot()
            if bot is None:
                print("Erreur critique à l'initialisation du bot. Arrêt.")
                return

            await bot.test_news_sentiment()

            # Analyse initiale du marché
            regime, _, _ = await bot.study_market("7d")
            log_dashboard(f"🔈 Régime de marché détecté: {regime}")

            # Boucle principale
            cycle = 0
            # ==== BOUCLE PRINCIPALE PATCHÉE POUR PAUSE ====
            while True:
                cycle += 1
                start = datetime.utcnow()

                bot.get_pending_sales()

                # === Gestion news pause manager ===
                try:
                    with open(bot.data_file, "r") as f:
                        shared_data = json.load(f)
                    news_sentiment = shared_data.get("sentiment", {})
                    news_list = news_sentiment.get("scores", [])
                except Exception:
                    news_list = []

                # Filtre les news non déjà traitées
                unprocessed_news = [n for n in news_list if not n.get("processed")]
                if bot.news_pause_manager.scan_news(unprocessed_news):
                    print("🚨 Pause trading à cause d'une news critique !")
                    for n in unprocessed_news:
                        n["processed"] = True
                    try:
                        with open(bot.data_file, "r") as f:
                            shared_data = json.load(f)
                    except Exception:
                        shared_data = {}
                    # PATCH: Sauvegarde fiable des news marquées comme traitées (processed: True)
                    if "sentiment" not in shared_data or not isinstance(
                        shared_data["sentiment"], dict
                    ):
                        shared_data["sentiment"] = {}
                    bot.safe_update_shared_data(
                        {
                            "sentiment": {
                                **shared_data.get("sentiment", {}),
                                "scores": news_list,
                            }
                        },
                        bot.data_file,
                    )

                # --- CORRECTION : Vérification de la pause globale via l'attribut ---
                trading_paused = bot.news_pause_manager.global_cycles_remaining > 0

                if trading_paused:
                    print(
                        "Trading en pause: calculs et signaux mis à jour, EXÉCUTION DES TRADES BLOQUÉE."
                    )

                try:
                    print(f"\n🔄 Cycle {cycle} - {start.strftime('%H:%M:%S')}")
                    # Hot reload IA
                    bot.check_reload_dl_model()

                    # Déclenchement stop-loss SPOT
                    for symbol, pos in list(bot.positions.items()):
                        if bot.is_long(symbol) and bot.check_stop_loss(symbol):
                            print(
                                f"[STOPLOSS] Déclenchement automatique du stop-loss pour {symbol}"
                            )
                            await bot.execute_trade(symbol, "SELL", pos["amount"])

                    # TP partiels et trailing TP sur toutes les positions longues
                    for symbol, pos in list(bot.positions.items()):
                        if pos.get("side") != "long":
                            continue
                        if "filled_tp_targets" not in pos:
                            pos["filled_tp_targets"] = [False, False]
                        if "price_history" not in pos:
                            pos["price_history"] = [pos["entry_price"]]
                        if "max_price" not in pos:
                            pos["max_price"] = pos["entry_price"]
                        last_price = None
                        if hasattr(bot, "ws_collector"):
                            last_price = bot.ws_collector.get_last_price(symbol)
                        if (
                            last_price is None
                            and symbol in bot.market_data
                            and "1h" in bot.market_data[symbol]
                        ):
                            closes = bot.market_data[symbol]["1h"].get("close", [])
                            if closes:
                                last_price = closes[-1]
                        if last_price is None:
                            continue
                        pos["price_history"].append(last_price)
                        # TP partiels
                        to_exit, new_filled = bot.exit_manager.check_tp_partial(
                            pos["entry_price"], last_price, pos["filled_tp_targets"]
                        )
                        if to_exit > 0 and pos["amount"] > 0:
                            amount_to_sell = pos["amount"] * to_exit
                            await bot.execute_trade(symbol, "SELL", amount_to_sell)
                            pos["amount"] -= amount_to_sell
                            pos["filled_tp_targets"] = new_filled
                            if pos["amount"] <= 0:
                                bot.positions.pop(symbol)
                                continue
                        # Trailing stop
                        should_exit, new_max = bot.exit_manager.check_trailing(
                            pos["entry_price"],
                            pos["price_history"],
                            pos.get("max_price", pos["entry_price"]),
                        )
                        pos["max_price"] = new_max
                        if should_exit and pos["amount"] > 0:
                            await bot.execute_trade(symbol, "SELL", pos["amount"])
                            bot.positions.pop(symbol)

                    # Déclenchement stop-loss et trailing stop SHORT BingX
                    for symbol, pos in list(bot.positions.items()):
                        if bot.is_short(symbol):
                            try:
                                symbol_bingx = symbol.replace("USDC", "USDT") + ":USDT"
                                ticker = await bot.bingx_client.fetch_ticker(
                                    symbol_bingx
                                )
                                price = float(ticker["last"])
                            except Exception:
                                continue
                            if bot.check_short_stop(
                                symbol, price=price, trailing_pct=0.03
                            ):
                                print(
                                    f"[SHORT STOP] Fermeture short {symbol} (prix: {price})"
                                )
                                await bot.telegram.send_message(
                                    f"🔴 <b>STOP SHORT déclenché</b>\n"
                                    f"Pair: {symbol}\n"
                                    f"Prix actuel: {price}\n"
                                    f"Position couverte automatiquement (stop/trailing stop)"
                                )
                                await bot.execute_trade(symbol, "BUY", pos["amount"])

                    # --- Analyse de marché et signaux, TOUJOURS exécuté ---
                    trade_decisions, regime = await execute_trading_cycle(
                        bot, valid_pairs
                    )

                    # Mise à jour des données du bot
                    bot.current_cycle = cycle
                    bot.regime = regime

                    # Calcul et stockage des indicateurs pour chaque paire/timeframe
                    bot.indicators = {}
                    for pair in bot.pairs_valid:
                        pair_key = pair.replace("/", "").upper()
                        for tf in bot.config["TRADING"]["timeframes"]:
                            if (
                                pair_key in bot.market_data
                                and tf in bot.market_data[pair_key]
                            ):
                                trend = bot.calculate_trend(
                                    bot.market_data[pair_key][tf]
                                )
                                volatility = bot.calculate_volatility(
                                    bot.market_data[pair_key][tf]
                                )
                                volume_profile = bot.calculate_volume_profile(
                                    bot.market_data[pair_key][tf]
                                )
                                dominant_signal = bot.get_dominant_signal(pair, tf)
                                df = bot.ws_collector.get_dataframe(pair_key, tf)
                                indics = (
                                    bot.add_indicators(df)
                                    if df is not None and not df.empty
                                    else {}
                                )
                                tf_key = f"{tf} | {pair}"
                                bot.indicators[tf_key] = {
                                    "trend": {"trend_strength": trend},
                                    "volatility": {"current_volatility": volatility},
                                    "volume": {"volume_profile": volume_profile},
                                    "dominant_signal": dominant_signal,
                                    "ta": indics if indics else {},
                                }

                    # --- PATCH: Sauvegarde des pauses actives, portefeuille et scores de décision ---
                    bot.news_pause_manager.on_cycle_end()  # décrémente les cycles_left
                    active_pauses = bot.get_active_pauses()
                    print("[DEBUG PATCH] Pauses RAM après tick:", active_pauses)
                    bot.sync_positions_with_binance()

                    # Ajout des scores de décision - PATCH pour vrai mapping
                    td_dict = {}
                    for td in trade_decisions:
                        signals = td.get("signals", {})
                        print(f"[DEBUG SIGNALS DASHBOARD] {td['pair']} {signals}")
                        td_dict[td["pair"]] = {
                            "confidence": td.get("confidence"),
                            "action": td.get("action"),
                            "tech": signals.get("technical"),
                            "ai": signals.get("ai"),
                            "sentiment": signals.get("sentiment"),
                        }
                    # PATCH: Ajoute toutes les paires manquantes avec valeurs nulles
                    for pair in bot.pairs_valid:
                        if pair not in td_dict:
                            td_dict[pair] = {
                                "confidence": 0,
                                "action": "neutral",
                                "tech": 0,
                                "ai": 0,
                                "sentiment": 0,
                            }
                    bot.trade_decisions = td_dict
                    print("[DEBUG DASHBOARD EXPORT]", json.dumps(td_dict, indent=2))

                    # Puis sauvegarde tout dans le shared_data
                    try:
                        with open(bot.data_file, "r") as f:
                            shared_data = json.load(f)
                    except Exception:
                        shared_data = {}

                    bot.safe_update_shared_data(
                        {
                            "active_pauses": active_pauses,
                            "positions_binance": getattr(bot, "positions_binance", {}),
                            "trade_decisions": bot.trade_decisions,
                        },
                        bot.data_file,
                    )

                    # Sauvegarde de l'état du bot à chaque cycle
                    bot.save_shared_data()

                    # --- EXÉCUTION DES TRADES UNIQUEMENT SI PAS DE PAUSE ---
                    if not trading_paused:
                        await execute_trade_decisions(bot, trade_decisions)
                    else:
                        print(
                            "🚫 [PAUSE] Exécution des trades bloquée, signaux et IA à jour."
                        )

                    # Entraînement IA automatique
                    if cycle % 50 == 0:
                        print(
                            "=== Entraînement automatique IA sur toutes les paires/timeframes ==="
                        )
                        bot.train_cnn_lstm_on_all_live()

                    # Entraînement IA manuel (optionnel)
                    bot.train_cnn_lstm_on_all_live()
                    print(
                        "=== Entraînement MANUEL IA sur toutes les paires/timeframes ==="
                    )
                    duration = (datetime.utcnow() - start).total_seconds()
                    print(f"✅ Cycle terminé en {duration:.1f}s")

                    # Envoi des mises à jour et rapports Telegram
                    await send_cycle_reports(
                        bot, trade_decisions, cycle, regime, duration
                    )

                except Exception as e:
                    error_msg = f"⚠️ Erreur cycle {cycle}: {e}"
                    logger.error(error_msg)
                    await bot.telegram.send_message(error_msg)

                # Attente avant le prochain cycle
                await asyncio.sleep(30)

        except KeyboardInterrupt:
            await handle_shutdown(bot, "👋 Bot arrêté proprement")
        except Exception as e:
            await handle_shutdown(bot, f"💥 Erreur fatale: {e}")

    # Démarrage de la boucle principale
    await main()


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
        log_dashboard(
            f"[TRADE-DECISION] {pair} | Action: {final_action.upper()} | Confiance: {confidence:.2f} | Score: {combined_score:.4f} | Tech: {signal['confidence']:.2f} | AI: {data.get('ai_prediction', 0.5):.2f} | Sentiment: {data.get('sentiment',0):.2f}"
        )
        await bot.telegram.send_message(
            f"🔔 <b>Décision de Trade</b>\n"
            f"Pair: {pair}\n"
            f"Action: <b>{final_action.upper()}</b>\n"
            f"Confiance: {confidence:.2f}\n"
            f"Score global: {combined_score:.4f}\n"
            f"Tech: {signal['confidence']:.2f}\n"
            f"AI: {data.get('ai_prediction', 0.5):.2f}\n"
            f"Sentiment: {data.get('sentiment',0):.2f}"
        )
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
        log_dashboard(f"💹 {len(opportunities)} opportunités d'arbitrage détectées")
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
    """
    Exécute toutes les décisions de trade du cycle.
    Intègre la gestion avancée de pause news par asset/action.
    Vérifie que les news contiennent bien les champs "symbols" et "sentiment".
    """
    # Vérification et warning sur les news du cycle (depuis le dernier batch d'analyse)
    news_list = []
    try:
        # On récupère les dernières news utilisées pour la pause
        with open(bot.data_file, "r") as f:
            shared_data = json.load(f)
        news_sentiment = shared_data.get("sentiment", {})
        news_list = news_sentiment.get("scores", [])
    except Exception:
        news_list = []

    # Vérification des champs "symbols" et "sentiment" dans les news
    for news in news_list:
        if "symbols" not in news or not news["symbols"]:
            log_dashboard(
                f"[NEWS CHECK] ⚠️ News sans champ 'symbols': {news.get('title', '')[:80]}"
            )
        if "sentiment" not in news:
            log_dashboard(
                f"[NEWS CHECK] ⚠️ News sans champ 'sentiment': {news.get('title', '')[:80]}"
            )

    for decision in trade_decisions:
        pair = decision.get("pair")
        action = decision.get("action")
        confidence = decision.get("confidence", 0)
        amount = calculate_position_size(
            bot, decision
        )  # Utilise la fonction déjà présente

        # Log avant exécution
        log_dashboard(
            f"[EXECUTE TRADE] {pair} | Action: {action.upper()} | Amount: {amount} | Confidence: {confidence}"
        )

        # ----- PATCH : Gestion de la pause news avancée -----
        # On vérifie si la pause news s'applique à cette paire et action
        active_pauses = bot.news_pause_manager.get_active_pauses()
        if any(
            p.get("asset") == pair or p.get("asset") == "GLOBAL" for p in active_pauses
        ):
            log_dashboard(
                f"[NEWS PAUSE] Trade {action.upper()} sur {pair} bloqué (news critique/pause en cours)"
            )
            await bot.telegram.send_message(
                f"🚨 Trading {action.upper()} sur {pair} bloqué à cause d'une news critique !"
            )
            continue  # On skip ce trade

        # Exécution réelle
        trade_result = await bot.execute_trade(pair, action, amount)
        # Notification Telegram
        await send_trade_notification(bot, decision, trade_result, amount)


def save_best_params(best_params, path="config/best_hyperparams.json"):
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2)


async def run_automl_tuning(bot, mode="cnn_lstm"):
    """Lance une optimisation AutoML/Optuna complète (manuelle ou auto)"""
    print("🔬 Lancement AutoML/Optuna...")
    import time

    start = time.time()
    if mode == "cnn_lstm":
        from src.optimization.optuna_wrapper import tune_hyperparameters

        best_params = tune_hyperparameters()
        print("✅ Optuna tuning terminé. Meilleurs hyperparams:", best_params)
        save_best_params(best_params)  # <-- Sauvegarde automatique
    elif mode == "full":
        from src.optimization.optuna_wrapper import optimize_hyperparameters_full

        best_trials = optimize_hyperparameters_full()
        print("✅ Optuna full tuning terminé. Résumé:", best_trials)
        # Si besoin, tu peux aussi sauvegarder best_trials ici
    else:
        print("❌ Mode AutoML inconnu")
        return
    duration = time.time() - start
    print(f"Durée optimisation: {duration:.1f}s")
    # (Optionnel) Recharge config/model avec les meilleurs params
    # bot.reload_model(best_params) ou autre logique
    return best_params if mode == "cnn_lstm" else best_trials


def calculate_position_size(bot, decision):
    """
    Sizing intelligent et adaptatif basé sur :
    - Confiance du signal
    - Kelly Criterion
    - Mode SAFE
    - Protection Drawdown
    - NOUVEAU: Ajustement par corrélation
    """
    try:
        # --- Configuration de base ---
        balance = bot.get_performance_metrics().get("balance", 0)
        confidence = float(decision.get("confidence", 0.5))
        MIN_NOTIONAL = 5  # Minimum USDC

        # --- Sizing selon confiance ---
        if confidence > 0.7:
            risk_pct = 0.09  # 9% max
        elif confidence > 0.4:
            risk_pct = 0.04  # 4%
        else:
            risk_pct = 0.02  # 2%

        # --- Ajustement Kelly ---
        perf = bot.get_performance_metrics()
        win_rate = perf.get("win_rate", 0.55)
        profit_factor = perf.get("profit_factor", 1.7)
        kelly = kelly_criterion(win_rate, profit_factor)

        if kelly > 0:
            risk_pct = min(risk_pct + kelly * 0.5, 0.12)

        # --- NOUVEAU: Ajustement par corrélation ---
        pair = decision.get("pair")
        if pair:
            correlations = bot.calculate_correlation_matrix()
            corr_factor = max(
                [v for k, v in correlations.items() if pair in k], default=0.5
            )
            # Réduit le sizing si forte corrélation
            risk_pct *= 1 - corr_factor * 0.5

        # --- Mode SAFE ---
        try:
            with open(bot.data_file, "r") as f:
                data = json.load(f)

            recent_trades = data.get("trade_history", [])[-5:]
            losses = [t for t in recent_trades if t.get("pnl_usd", 0) < 0]
            wins = [t for t in recent_trades if t.get("pnl_usd", 0) > 0]

            mode_safe = len(losses) >= 3 and len(wins) == 0
            if mode_safe and wins:
                mode_safe = False

            bot.safe_update_shared_data({"safe_mode": mode_safe}, bot.data_file)

            if mode_safe:
                risk_pct *= 0.25
                print("[SAFE MODE] Sizing -75%")

        except Exception as e:
            print(f"[WARNING] Erreur mode safe: {e}")

        # --- Protection Drawdown ---
        try:
            with open(bot.data_file, "r") as f:
                data = json.load(f)

            equity_history = data.get("equity_history", [])
            if equity_history and len(equity_history) >= 30:
                balances = [pt["balance"] for pt in equity_history if "balance" in pt]
                peak = max(balances)
                trough = min(balances)
                drawdown = (trough - peak) / peak if peak > 0 else 0

                if drawdown < -0.15:
                    risk_pct *= 0.5
                    print(f"[DRAWDOWN] Sizing -50% (DD: {drawdown:.1%})")

        except Exception as e:
            print(f"[WARNING] Erreur drawdown: {e}")

        # --- Calcul final ---
        size = balance * risk_pct
        size = max(MIN_NOTIONAL, round(size, 2))

        print(f"[SIZING] {size:.2f} USDC ({risk_pct*100:.1f}% du capital)")
        return size

    except Exception as e:
        import logging

        logging.error(f"Erreur sizing: {e}")
        return MIN_NOTIONAL


async def send_trade_notification(bot, decision, trade_result, amount):
    """
    Envoie une notification Telegram centralisée et lisible pour un trade exécuté.
    Affiche tous les signaux clés et la confiance de la décision.
    """
    try:
        # Détermination de l'emoji selon l'action
        action = decision.get("action", "").lower()
        emoji = "🟢" if action == "buy" else "🔴" if action == "sell" else "⚪️"

        # Construction du message
        message = (
            f"{emoji} <b>TRADE EXÉCUTÉ</b>\n\n"
            f"📊 Paire : {decision.get('pair', '?')}\n"
            f"Action : <b>{action.upper()}</b>\n"
            f"Montant : {amount}\n"
            f"Prix : {trade_result.get('avg_price', 'N/A')}\n"
            f"Total : {float(amount) * float(trade_result.get('avg_price', 0)):.2f} USDT\n"
            f"Confiance : {decision.get('confidence', 0):.0%}\n"
            f"Signaux : Tech {decision.get('signals', {}).get('technical', 0):.0%} | "
            f"IA {decision.get('signals', {}).get('ai', 0):.2f} | "
            f"Sentiment {decision.get('signals', {}).get('sentiment', 0):.2f}\n"
        )
        await bot.telegram.send_message(message)

    except Exception as e:
        logging.error(f"Erreur envoi notification: {e}")


def build_telegram_summary(bot, trade_decisions, news_sentiment):
    summary = "🟢 <b>Résumé du cycle</b>\n"
    # Régime
    summary += f"📊 Régime de marché : {bot.regime}\n"
    # Paires principales (top 5)
    top_pairs = (
        ", ".join([d["pair"] for d in trade_decisions[:5]])
        if trade_decisions
        else "N/A"
    )
    summary += f"📈 Paires principales : {top_pairs}\n"
    # Décisions de trade principales (top 5)
    for d in trade_decisions[:5]:
        emoji = (
            "🟢" if d["action"] == "buy" else "🔴" if d["action"] == "sell" else "⚪️"
        )
        conf = int(d["confidence"] * 100)
        summary += f"{emoji} {d['pair']} : {d['action'].upper()} ({conf}%)\n"
    # News principales (top 3)
    if news_sentiment and "latest_news" in news_sentiment:
        summary += "\n📰 News principales :\n"
        for title in news_sentiment["latest_news"][:3]:
            summary += f"• {title}\n"
    return summary


async def send_cycle_reports(bot, trade_decisions, cycle, regime, duration):
    """
    Envoi des rapports de fin de cycle avec :
    - Résumé des trades
    - Analyse complète
    - Métriques avancées
    - Alertes de risque
    """
    import json
    import numpy as np
    from datetime import datetime

    try:
        # 1. Rapport des trades du cycle
        await send_trade_summary(bot, trade_decisions)

        # 2. Construction des analyses
        analysis_data = await prepare_analysis_data(bot, trade_decisions)

        # 3. Sauvegarde des données
        await save_cycle_data(bot, analysis_data)

        # 4. Envoi du rapport complet
        await send_analysis_report(bot, analysis_data)

        # 5. Alertes de risque avancées
        await check_risk_alerts(bot, analysis_data)

    except Exception as e:
        logging.error(f"Erreur envoi rapports: {e}")


async def send_trade_summary(bot, trade_decisions):
    """Envoi du résumé des trades"""
    if trade_decisions:
        trade_report = "💹 <b>Résumé des trades du cycle</b>\n\n"
        for trade in trade_decisions:
            emoji = (
                "🟢"
                if trade["action"] == "buy"
                else "🔴" if trade["action"] == "sell" else "⚪️"
            )
            pair = trade.get("pair", "INCONNU")
            signals = trade.get("signals", {})

            # Utilisation de format strings pour plus de clarté
            trade_report += (
                f"{emoji} {pair}: {trade['action'].upper()} "
                f"({trade.get('confidence', 0):.0%}) | "
                f"Tech {signals.get('technical', 0):.0%} | "
                f"IA {signals.get('ai', 0):.2f} | "
                f"Sent {signals.get('sentiment', 0):.2f}\n"
            )
        await bot.telegram.send_message(trade_report)
    else:
        await bot.telegram.send_cycle_update(cycle, regime, duration)


async def prepare_analysis_data(bot, trade_decisions):
    """Préparation des données d'analyse"""
    # Construction des analyses par timeframe/paire
    indicators_analysis = {}
    trade_decisions_dict = {}

    # Analyse des indicateurs
    for pair in bot.pairs_valid:
        pair_key = pair.replace("/", "").upper()
        for tf in bot.config["TRADING"]["timeframes"]:
            tf_key = f"{tf} | {pair}"
            indics = bot.indicators.get(pair_key, {}).get(tf, {})
            indicators_analysis[tf_key] = indics if indics else {}

    # Organisation des décisions de trade
    for decision in trade_decisions:
        tf = decision.get("tf", "1h")
        pair = decision.get("pair", "")
        tf_key = f"{tf} | {pair}"

        trade_decisions_dict[tf_key] = {
            "pair": pair,
            "tf": tf,
            "action": decision.get("action", "NEUTRAL").upper(),
            "confidence": decision.get("confidence", 0),
            "tech": decision.get("signals", {}).get("technical", 0),
            "ai": decision.get("signals", {}).get("ai", 0),
            "sentiment": decision.get("signals", {}).get("sentiment", 0),
        }

    # Métriques avancées (intégration de track_advanced_metrics)
    advanced_metrics = bot.track_advanced_metrics()

    return {
        "indicators": indicators_analysis,
        "decisions": trade_decisions_dict,
        "metrics": advanced_metrics,
        "perf": bot.get_performance_metrics(),
    }


async def save_cycle_data(bot, analysis_data):
    """Sauvegarde des données du cycle"""
    try:
        with open(bot.data_file, "r") as f:
            data = json.load(f)

        # Mise à jour des données
        equity_history = data.get("equity_history", [])
        equity_history.append(
            {
                "timestamp": get_current_time(),
                "balance": analysis_data["perf"].get("balance", 0),
                "metrics": analysis_data["metrics"],  # Ajout des métriques avancées
            }
        )

        # Limitation de l'historique
        if len(equity_history) > 1000:
            equity_history = equity_history[-1000:]

        # Sauvegarde sécurisée
        bot.safe_update_shared_data(
            {
                "trade_decisions": analysis_data["decisions"],
                "equity_history": equity_history,
                "positions_binance": getattr(bot, "positions_binance", {}),
                "advanced_metrics": analysis_data["metrics"],
            },
            bot.data_file,
        )

    except Exception as e:
        logging.error(f"Erreur sauvegarde données: {e}")


async def check_risk_alerts(bot, analysis_data):
    """Vérification et envoi des alertes de risque"""
    try:
        equity_history = analysis_data.get("equity_history", [])
        perf = analysis_data["perf"]

        # 1. Alerte Kelly
        kelly = kelly_criterion(
            win_rate=perf.get("win_rate", 0), payoff_ratio=perf.get("profit_factor", 1)
        )
        if abs(kelly) > 0.5:
            await bot.telegram.send_message(
                f"⚠️ Kelly fraction élevée: {kelly:.2f}\n"
                f"Réduction recommandée du sizing!"
            )

        # 2. Alerte Drawdown
        equity_curve = [
            pt.get("balance", 0) for pt in equity_history if pt.get("balance", 0) > 0
        ]
        if equity_curve and len(equity_curve) > 10:
            max_dd = calculate_max_drawdown(np.array(equity_curve))
            if max_dd < -0.15:
                await bot.telegram.send_message(
                    f"🚨 Drawdown critique: {max_dd:.2%}\n"
                    f"Actions recommandées:\n"
                    f"- Réduction du sizing\n"
                    f"- Pause trading conseillée"
                )

        # 3. Alerte VaR
        if len(equity_curve) > 10:
            try:
                equity_curve_np = np.array(equity_curve)
                returns = np.diff(equity_curve_np) / equity_curve_np[:-1]
                var95 = calculate_var(returns, 0.05)
                if var95 < -0.05:
                    await bot.telegram.send_message(
                        f"🛑 VaR(95%) critique: {var95:.2%}\n"
                        f"Risque de perte important!"
                    )
            except Exception:
                pass

    except Exception as e:
        logging.error(f"Erreur alertes risque: {e}")


async def handle_shutdown(bot, message):
    """Gère l'arrêt propre du bot"""
    try:
        print(f"\n{message}")
        await bot.telegram.send_message(message)
        await bot.ws_collector.stop()
        bot.save_shared_data()
    except Exception as e:
        logging.error(f"Erreur arrêt bot: {e}")


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    batch_size = int(batch_size)  # <-- PATCH FONDAMENTAL
    n_epochs = 5
    pairs = load_config()
    scores = []
    for pair in pairs:
        try:
            print(f"[Optuna] TRAIN sur {pair}…")
            model = HybridAI(
                pair=pair,
                window=30,
                interval="1h",
                start_str="1 Jan, 2023",
                end_str="now",
                cache_dir="data_cache",
            )
            acc = model.validate(lr=lr, batch_size=batch_size, n_epochs=n_epochs)
            print(f"[Optuna] {pair} | Accuracy={acc:.4f}")
            scores.append(acc)
        except Exception as e:
            print(f"[Optuna] Erreur sur {pair}: {e}")
    if not scores:
        print("[Optuna] Aucune paire dispo pour ce trial !")
    return float(sum(scores)) / len(scores) if scores else 0.0


if __name__ == "__main__":

    # --- 1. Argument parsing avancé
    import argparse

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
    parser.add_argument("--capital", type=float, default=0, help="Capital initial")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sma",
        choices=["sma", "breakout", "arbitrage"],
        help="Stratégie à utiliser",
    )
    parser.add_argument(
        "--auto-strategy",
        action="store_true",
        help="Active l'auto-stratégie (recherche + utilisation)",
    )
    parser.add_argument(
        "--auto-pair",
        type=str,
        default="BTCUSDT",
        help="Paire à utiliser pour l'auto-stratégie",
    )
    parser.add_argument(
        "--auto-timeframe",
        type=str,
        default="1h",
        help="Timeframe à utiliser pour l'auto-stratégie",
    )
    parser.add_argument(
        "--auto-days",
        type=int,
        default=30,
        help="Nombre de jours d'historique pour l'auto-stratégie",
    )
    parser.add_argument(
        "--auto-n", type=int, default=50, help="Nombre de stratégies à générer/tester"
    )
    parser.add_argument(
        "--optuna-signal-fusion",
        action="store_true",
        help="Lance l'optimisation AutoML des pondérations de signaux",
    )
    args, unknown = parser.parse_known_args()

    # --- 2. Mode AutoML/Tuning (prioritaire sur tout le reste)
    if "automl" in sys.argv or "tune" in sys.argv:
        asyncio.run(run_automl_tuning(None, mode="cnn_lstm"))

    # --- 2bis. Mode Optuna signal fusion
    elif args.optuna_signal_fusion:
        print("=== Lancement Optuna signal fusion (diagnostic print) ===")
        from src.optimization.signal_fusion_automl import optimize_signal_fusion_and_mm

        optimize_signal_fusion_and_mm(n_trials=100)
        exit(0)

    # --- 3. Mode auto-strategy (AUTO-ML stratégies)
    elif "auto-strategy" in sys.argv:
        # Paramètres pour Binance
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        symbol = args.auto_pair.upper()
        tf_str = args.auto_timeframe.lower()
        interval = getattr(Client, f"KLINE_INTERVAL_{tf_str.upper()}")
        nb_days = args.auto_days

        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=nb_days)
        start_str = start_dt.strftime("%d %b %Y")
        end_str = end_dt.strftime("%d %b %Y")

        # Récupère les données Binance
        df = fetch_binance_ohlcv(
            symbol,
            interval,
            start_str,
            end_str,
            api_key=api_key,
            api_secret=api_secret,
        )
        if df is None or len(df) == 0:
            print("Aucune donnée récupérée sur Binance, impossible d’auto-stratégie.")
            sys.exit(1)

        df.columns = [col.lower() for col in df.columns]  # Sécurité
        best_config, best_score = auto_generate_and_backtest(df, n_strats=args.auto_n)
        print("Meilleure stratégie trouvée :", best_config)
        print("Score (profit brut sur l'historique):", best_score)

        # Sauvegarde pour usage live
        if not os.path.exists("config"):
            os.makedirs("config", exist_ok=True)
        with open("config/auto_strategy.json", "w") as f:
            json.dump(
                {
                    "pair": symbol,
                    "timeframe": tf_str,
                    "config": best_config,
                    "score": best_score,
                    "date": datetime.utcnow().isoformat(),
                },
                f,
                indent=4,
            )

        # Envoi rapport Telegram
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            from src.bot_runner import TelegramNotifier, get_current_time, CURRENT_USER
            import asyncio

            notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            rapport = (
                f"🔬 <b>Auto-Strategy Report</b>\n\n"
                f"Paire: <b>{symbol}</b>\nTimeframe: <b>{tf_str}</b>\n"
                f"Meilleure config trouvée : <code>{best_config}</code>\n"
                f"Score (profit brut): <b>{best_score:.2f}</b>\n"
                f"Date: {get_current_time()}\n"
                f"Utilisateur: {CURRENT_USER}"
            )
            asyncio.run(notifier.send_message(rapport))

        sys.exit(0)

    # --- 4. Mode backtest CLI
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

    # --- 5. Entraînement IA live
    elif "train-cnn-lstm" in sys.argv:
        bot = TradingBotM4()
        # Préchargement historique pour chaque paire/timeframe avant entraînement IA
        if hasattr(bot, "ws_collector") and hasattr(bot, "binance_client"):
            for symbol in bot.pairs_valid:
                symbol_binance = symbol.replace("/", "").upper()
                for tf in bot.config["TRADING"]["timeframes"]:
                    try:
                        bot.ws_collector.preload_historical(
                            bot.binance_client, symbol_binance, tf, limit=2000
                        )
                        print(f"Préchargement {symbol_binance} {tf} OK")
                    except Exception as e:
                        print(f"Erreur préchargement {symbol_binance} {tf} : {e}")
        # Lancement de l'entraînement IA sur les données chargées
        bot.train_cnn_lstm_on_all_live()
        sys.exit(0)

    # --- 6. Lancement du bot de trading en mode normal
    else:
        asyncio.run(run_clean_bot())
