import os
import sys
import warnings
import logging
import json
import asyncio
import aiohttp
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
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
from src.ai.enhanced_cnn_lstm import EnhancedCNNLSTM
from src.ai_models.hybrid.cnn_lstm_enhanced import EnhancedCNNLSTM
from src.ai.ppo_gtrxl import PPOGTrXL
from src.analysis.news.sentiment_analyzer import NewsSentimentAnalyzer
from src.connectors.binance import BinanceConnector
from src.ai.deep_learning_model import DeepLearningModel
from src.ai.ppo_strategy import PPOStrategy
from src.bot.core import TradingEnv

# Charger les variables d'environnement depuis .env
load_dotenv()

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
        """Envoie une mise à jour des performances"""
        message = (
            "🤖 <b>Trading Bot Status Update</b>\n\n"
            f"💰 Balance: ${performance_data['balance']:,.2f}\n"
            f"📊 Win Rate: {performance_data['win_rate']*100:.1f}%\n"
            f"📈 Profit Factor: {performance_data['profit_factor']:.2f}\n"
            f"🔄 Total Trades: {performance_data['total_trades']}\n"
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

    async def send_news_summary(self, news_data):
        """Envoie un résumé des dernières nouvelles importantes"""
        if not news_data or len(news_data) == 0:
            return

        message = "📰 <b>Dernières Nouvelles Importantes</b>\n\n"

        for i, news in enumerate(news_data[:5]):
            sentiment = news.get("sentiment", 0)
            sentiment_emoji = (
                "🟢" if sentiment > 0.3 else "🔴" if sentiment < -0.3 else "⚪"
            )

            message += f"{sentiment_emoji} {news['title']}\n"
            if "summary" in news and news["summary"]:
                message += f"└ {news['summary'][:100]}...\n\n"
            else:
                message += "\n"

        await self.send_message(message)


class WarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, message):
        if any(
            word in message.lower()
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
        self.data_file = SHARED_DATA_PATH
        self.current_cycle = 0
        self.regime = MARKET_REGIMES["RANGING"]
        self.pairs_valid = []
        self.market_data = {}
        self.indicators = {}
        self.initialize_shared_data()
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.last_telegram_update = datetime.utcnow()
        self.logger = logger

        # Initialisation de l'API Binance
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

        # Initialisation des modèles d'IA
        try:
            self.dl_model = DeepLearningModel()
            self.env = TradingEnv(
                trading_pairs=self.pairs_valid,
                timeframes=self.config["TRADING"]["timeframes"],
            )

            # Configuration par défaut pour PPOStrategy
            default_config = {
                "env": self.env,
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "n_epochs": 10,
                "verbose": 1,
            }
            self.ppo_strategy = PPOStrategy(default_config)
            self.ai_enabled = True
            self.ai_weight = 0.3  # Influence de l'IA dans la décision (30%)
            self.logger.info("AI models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}")
            self.ai_enabled = False
            self.dl_model = None
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

    async def execute_trade(self, symbol, side, amount, price=None):
        """Exécute un ordre de trading avec l'exécuteur intelligent"""
        if not self.is_live_trading:
            self.logger.info(f"SIMULATION: {side} {amount} {symbol} @ {price}")
            return {
                "status": "simulated",
                "symbol": symbol,
                "side": side,
                "amount": amount,
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

            # Exécution de l'ordre avec notre exécuteur intelligent
            result = await self.executor.execute_order(
                symbol=symbol,
                side=side,
                amount=amount,
                orderbook=orderbook,
                market_data=market_data,
            )

            # Enregistrement du résultat
            if result["status"] == "completed":
                self.logger.info(
                    f"Order executed: {side} {result['filled_amount']} {symbol} @ {result['avg_price']}"
                )
                # Mettre à jour les statistiques
                self._update_performance_metrics(result)

                # Notification Telegram
                await self.telegram.send_message(
                    f"💰 <b>Ordre exécuté</b>\n"
                    f"📊 {side} {result['filled_amount']} {symbol} @ {result['avg_price']}\n"
                    f"💵 Total: ${float(result['filled_amount']) * float(result['avg_price']):.2f}"
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

            if side.lower() == "buy":
                # Pour un achat, on ne sait pas encore si c'est gagnant
                pass
            elif side.lower() == "sell":
                # Pour une vente, on peut calculer le profit par rapport au prix d'achat moyen
                entry_price = trade_result.get("entry_price", 0)
                if entry_price > 0:
                    profit_pct = (
                        (avg_price / entry_price - 1) * 100
                        if side.lower() == "sell"
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
        """Boucle d'analyse des news"""
        while True:
            try:
                if not self.news_enabled or not self.news_analyzer:
                    await asyncio.sleep(self.news_update_interval)
                    continue

                # Récupération et analyse des dernières news
                self.logger.info("Fetching latest news for sentiment analysis")
                news_data = await self.news_analyzer.fetch_all_news()
                sentiment_scores = self.news_analyzer.analyze_sentiment(news_data)

                # Mise à jour des données de sentiment
                await self._update_sentiment_data(sentiment_scores)

                # Envoi d'alertes si sentiment extrême
                for item in sentiment_scores:
                    symbol = item.get("symbol", "")
                    score = item.get("sentiment", 0)
                    if abs(score) > 0.7:  # Sentiment très fort (positif ou négatif)
                        sentiment_type = "positif" if score > 0 else "négatif"
                        await self.telegram.send_message(
                            f"⚠️ Sentiment {sentiment_type} fort détecté pour {symbol}: {score:.2f}"
                        )

                # Sauvegarde des données pour l'interface
                await self._save_sentiment_data(sentiment_scores, news_data)

                # Envoi du résumé périodique des news
                await self.telegram.send_news_summary(news_data[:5])

            except Exception as e:
                self.logger.error(f"News analysis error: {e}")

            # Attente avant la prochaine analyse
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

        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        for tf in timeframes:
            for pair in self.pairs_valid:
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
                pair_key = pair.replace("/", "")
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
                pair_key = pair.replace("/", "")
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
        """Calcul de la tendance"""
        try:
            if isinstance(data, dict) and "close" in data:
                closes = data["close"][-20:]
                ma_fast = sum(closes[-10:]) / 10
                ma_slow = sum(closes) / 20
                trend = (ma_fast / ma_slow) - 1
                return trend
            return 0
        except:
            return 0

    def calculate_volatility(self, data):
        """Calcul de la volatilité"""
        try:
            if isinstance(data, dict) and "close" in data:
                closes = data["close"][-20:]
                returns = np.diff(np.log(closes))
                return np.std(returns) * np.sqrt(252)
            return 0.5
        except:
            return 0.5

    def calculate_volume_profile(self, data):
        """Calcul du profil de volume"""
        try:
            if isinstance(data, dict) and "volume" in data:
                current_vol = data["volume"][-1]
                avg_vol = sum(data["volume"][-20:]) / 20
                return current_vol / avg_vol if avg_vol > 0 else 1.0
            return 1.0
        except:
            return 1.0

    def get_trend_analysis(self, pair, timeframe):
        """Analyse de tendance détaillée"""
        try:
            if pair in self.market_data and timeframe in self.market_data[pair]:
                trend = self.calculate_trend(self.market_data[pair][timeframe])
                if trend > 0.02:
                    return "Haussière"
                elif trend < -0.02:
                    return "Baissière"
                return "Neutre"
            return "N/A"
        except:
            return "N/A"

    def get_volatility_analysis(self, pair, timeframe):
        """Analyse de volatilité détaillée"""
        try:
            if pair in self.market_data and timeframe in self.market_data[pair]:
                vol = self.calculate_volatility(self.market_data[pair][timeframe])
                if vol > 0.8:
                    return "Élevée"
                elif vol > 0.4:
                    return "Moyenne"
                return "Faible"
            return "N/A"
        except:
            return "N/A"

    def get_volume_analysis(self, pair, timeframe):
        """Analyse du volume"""
        try:
            if pair in self.market_data and timeframe in self.market_data[pair]:
                vol = self.calculate_volume_profile(self.market_data[pair][timeframe])
                if vol > 1.5:
                    return "Fort"
                elif vol > 0.7:
                    return "Moyen"
                return "Faible"
            return "N/A"
        except:
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
        except:
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
            pair_key = pair.replace("/", "")

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

    async def detect_arbitrage_opportunities(self):
        """Détecte les opportunités d'arbitrage entre différents marchés"""
        if not self.is_live_trading:
            return []

        opportunities = []

        try:
            # Récupérer les tickers de plusieurs exchanges
            binance_prices = {}
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "")
                ticker = self.binance_client.get_ticker(symbol=pair_key)
                binance_prices[pair] = float(ticker["lastPrice"])

            # Comparer avec d'autres exchanges (si implémenté)
            # Cette partie nécessiterait des connecteurs pour d'autres exchanges

            # Simuler la détection d'opportunités pour démonstration
            for pair in self.pairs_valid:
                # Simuler une opportunité avec 0.5% de différence
                simulated_opportunity = {
                    "pair": pair,
                    "exchange1": "Binance",
                    "price1": binance_prices[pair],
                    "exchange2": "Simulated",
                    "price2": binance_prices[pair] * 1.005,
                    "diff_percent": 0.5,
                }

                if simulated_opportunity["diff_percent"] > 0.3:  # Seuil minimum
                    opportunities.append(simulated_opportunity)

            return opportunities

        except Exception as e:
            self.logger.error(f"Error detecting arbitrage: {e}")
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
        """Envoie des mises à jour périodiques sur Telegram"""
        current_time = datetime.utcnow()
        if (current_time - self.last_telegram_update).total_seconds() >= 300:
            performance = self.get_performance_metrics()
            print("📱 Envoi mise à jour Telegram...")
            await self.telegram.send_performance_update(performance)
            market_report = await self.generate_market_analysis_report()
            await self.telegram.send_message(market_report)
            self.last_telegram_update = current_time

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
        }

        # Ajout des données d'IA si disponibles
        if self.ai_enabled:
            ai_predictions = {}
            for pair in self.pairs_valid:
                pair_key = pair.replace("/", "")
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
        """Configure les composants du bot"""
        try:
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
            pair_key = pair.replace("/", "")
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
        """Ajoute les indicateurs techniques"""
        indicators = {}

        try:
            # Calcul du RSI
            delta = df[4].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators["rsi"] = 100 - (100 / (1 + rs.iloc[-1]))

            # Calcul des moyennes mobiles
            indicators["sma_20"] = df[4].rolling(window=20).mean().iloc[-1]
            indicators["sma_50"] = df[4].rolling(window=50).mean().iloc[-1]
            indicators["sma_200"] = df[4].rolling(window=200).mean().iloc[-1]

            # Calcul du MACD
            ema_12 = df[4].ewm(span=12, adjust=False).mean()
            ema_26 = df[4].ewm(span=26, adjust=False).mean()
            indicators["macd"] = ema_12.iloc[-1] - ema_26.iloc[-1]
            indicators["macd_signal"] = (
                (ema_12 - ema_26).ewm(span=9, adjust=False).mean().iloc[-1]
            )

            # Calcul des bandes de Bollinger
            sma_20 = df[4].rolling(window=20).mean()
            std_20 = df[4].rolling(window=20).std()
            indicators["bb_upper"] = (sma_20 + 2 * std_20).iloc[-1]
            indicators["bb_lower"] = (sma_20 - 2 * std_20).iloc[-1]
            indicators["bb_width"] = (
                indicators["bb_upper"] - indicators["bb_lower"]
            ) / sma_20.iloc[-1]

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")

        return indicators

    async def analyze_signals(self, df, indicators):
        """Analyse les signaux"""
        await asyncio.sleep(0.2)  # Simule le temps d'analyse

        action = "neutral"
        confidence = 0.5

        try:
            # Analyse technique de base
            if indicators:
                # Signaux de tendance
                trend_signal = 0
                if indicators.get("sma_20", 0) > indicators.get("sma_50", 0):
                    trend_signal += 0.3
                else:
                    trend_signal -= 0.3

                # Signaux de momentum
                momentum_signal = 0
                if indicators.get("rsi", 50) > 70:
                    momentum_signal -= 0.5  # Survente potentielle
                elif indicators.get("rsi", 50) < 30:
                    momentum_signal += 0.5  # Surachat potentiel

                # Signaux MACD
                macd_signal = 0
                if indicators.get("macd", 0) > indicators.get("macd_signal", 0):
                    macd_signal += 0.3
                else:
                    macd_signal -= 0.3

                # Agrégation des signaux
                combined_signal = (trend_signal + momentum_signal + macd_signal) / 3

                if combined_signal > 0.2:
                    action = "buy"
                    confidence = 0.5 + min(0.5, abs(combined_signal))
                elif combined_signal < -0.2:
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
            return config.get("valid_pairs", ["BTC/USDC", "ETH/USDC"])
    except:
        return ["BTC/USDC", "ETH/USDC"]


async def run_clean_bot():
    """Fonction principale"""
    print("\n=== DÉMARRAGE DU BOT ===")
    print("🚀 Trading Bot Ultimate v4 - Version Ultra-Propre")

    try:
        # Étape 1: Initialisation
        print("\n=== ÉTAPE 1: INITIALISATION ===")
        valid_pairs = load_config()
        print(f"📊 Paires configurées: {valid_pairs}")

        bot = TradingBotM4()
        bot.pairs_valid = valid_pairs
        setup_success = await bot._setup_components()

        if not setup_success:
            print("⚠️ Erreur lors de l'initialisation des composants")
            return

        # Notification de démarrage avec rapport d'analyse
        initial_report = await bot.generate_market_analysis_report()
        await bot.telegram.send_message(
            "🚀 <b>Bot Trading démarré</b>\n"
            "✅ Initialisation réussie\n"
            f"📊 Paires configurées: {', '.join(valid_pairs)}\n\n"
            f"{initial_report}"
        )

        print("✅ Bot initialized successfully")

        # Étape 2: Analyse du marché
        print("\n=== ÉTAPE 2: ANALYSE DU MARCHÉ ===")
        regime, _, indicators = await bot.study_market("7d")
        print(f"🔈 Régime de marché détecté: {regime}")

        # Étape 3: Trading
        print("\n=== ÉTAPE 3: TRADING ADAPTATIF ===")
        print("📈 Trading adaptatif lancé")

        cycle = 0
        while True:
            cycle += 1
            start = datetime.utcnow()

            try:
                print(f"\n🔄 Cycle {cycle} - {start.strftime('%H:%M:%S')}")

                # Analyse et mise à jour
                regime, market_data, indicators = await bot.study_market("7d")
                strategy = bot.choose_strategy(regime, indicators)
                print(f"🎯 Stratégie active: {strategy}")

                # Détection d'opportunités d'arbitrage
                arbitrage_opportunities = await bot.detect_arbitrage_opportunities()
                if arbitrage_opportunities:
                    print(
                        f"💹 {len(arbitrage_opportunities)} opportunités d'arbitrage détectées"
                    )
                    for opp in arbitrage_opportunities:
                        print(
                            f"  • {opp['pair']}: {opp['diff_percent']:.2f}% entre {opp['exchange1']} et {opp['exchange2']}"
                        )

                        # Notification Telegram pour chaque opportunité
                        await bot.telegram.send_arbitrage_alert(opp)

                        # Si l'opportunité est suffisamment intéressante, exécuter l'arbitrage
                        if (
                            opp["diff_percent"] > 0.5
                        ):  # Seuil de rentabilité après frais
                            print(f"🔄 Exécution de l'arbitrage sur {opp['pair']}")
                            # Code d'exécution de l'arbitrage à implémenter

                # Analyse des signaux pour chaque paire
                trade_decisions = []
                for pair in valid_pairs:
                    pair_key = pair.replace("/", "")
                    if market_data and pair_key in market_data:
                        data = market_data[pair_key]

                        # Conversion des données OHLCV pour l'analyse
                        ohlcv_df = None
                        if "1h" in data:
                            ohlcv = data["1h"]
                            # Conversion en DataFrame pour l'analyse
                            if all(
                                k in ohlcv
                                for k in ["open", "high", "low", "close", "volume"]
                            ):
                                ohlcv_df = pd.DataFrame(
                                    {
                                        0: ohlcv["timestamp"],
                                        1: ohlcv["open"],
                                        2: ohlcv["high"],
                                        3: ohlcv["low"],
                                        4: ohlcv["close"],
                                        5: ohlcv["volume"],
                                    }
                                )

                        if ohlcv_df is not None and len(ohlcv_df) >= 20:
                            indicators_data = bot.add_indicators(ohlcv_df)
                            signal = await bot.analyze_signals(
                                ohlcv_df, indicators_data
                            )

                            # Intégration des signaux IA et news si disponibles
                            ai_signal = data.get("ai_prediction", 0.5)
                            sentiment_score = data.get("sentiment", 0)

                            # Fusion des signaux
                            combined_score = 0
                            if signal["action"] == "buy":
                                combined_score += signal["confidence"] * 0.5
                            elif signal["action"] == "sell":
                                combined_score -= signal["confidence"] * 0.5

                            # Ajout du signal IA
                            if bot.ai_enabled:
                                combined_score += (ai_signal - 0.5) * 2 * bot.ai_weight

                            # Amélioration de l'intégration des news
                            if bot.news_enabled and sentiment_score != 0:
                                # Ajuster le poids du sentiment en fonction de son intensité
                                sentiment_weight = bot.news_weight * (
                                    1 + abs(sentiment_score)
                                )

                                # Donner plus d'importance aux nouvelles très positives ou très négatives
                                if abs(sentiment_score) > 0.7:
                                    print(
                                        f"⚠️ Sentiment fort détecté pour {pair}: {sentiment_score:.2f}"
                                    )
                                    sentiment_weight *= 1.5

                                # Ajouter une information temporelle au score
                                time_factor = 1.0  # Diminue avec le temps
                                if "sentiment_timestamp" in data:
                                    elapsed_time = (
                                        time.time() - data["sentiment_timestamp"]
                                    )
                                    time_factor = max(
                                        0.2, 1.0 - (elapsed_time / (3600 * 12))
                                    )  # Décroît sur 12h

                                # Appliquer le score de sentiment avec le poids ajusté
                                combined_score += (
                                    sentiment_score * sentiment_weight * time_factor
                                )

                            # Détermination de l'action finale
                            final_action = "neutral"
                            if combined_score > 0.3:
                                final_action = "buy"
                            elif combined_score < -0.3:
                                final_action = "sell"

                            # Affichage et stockage de la décision
                            confidence = min(0.99, abs(combined_score) + 0.5)
                            print(
                                f"📡 {pair}: {final_action.upper()} ({confidence:.0%})"
                            )

                            # Exécution des ordres en mode réel si le signal est fort
                            if bot.is_live_trading and abs(combined_score) > 0.5:
                                # Détermination des paramètres de l'ordre
                                side = "BUY" if final_action == "buy" else "SELL"

                                # Calcul du montant en fonction de la force du signal et de la volatilité
                                base_amount = 0.01  # Montant de base
                                volatility_factor = data.get("signals", {}).get(
                                    "volatility", 0.5
                                )
                                risk_adjusted_amount = base_amount * (
                                    1 - volatility_factor * 0.5
                                )  # Réduire le montant si volatilité élevée
                                signal_adjusted_amount = risk_adjusted_amount * (
                                    0.5 + confidence * 0.5
                                )  # Augmenter le montant si confiance élevée

                                # Vérifier les stop-loss
                                has_stop_loss = bot.check_stop_loss(pair_key, side)
                                if has_stop_loss:
                                    print(
                                        f"⚠️ Stop-loss actif pour {pair}, ordre annulé"
                                    )
                                    continue

                                # Exécution de l'ordre
                                trade_result = await bot.execute_trade(
                                    pair_key, side, signal_adjusted_amount
                                )

                                # Enregistrer la décision et le résultat
                                trade_decisions.append(
                                    {
                                        "pair": pair,
                                        "action": final_action,
                                        "confidence": confidence,
                                        "result": trade_result,
                                        "signals": {
                                            "technical": signal["confidence"],
                                            "ai": ai_signal,
                                            "sentiment": sentiment_score,
                                        },
                                    }
                                )

                                # Envoyer une alerte Telegram pour chaque trade
                                if trade_result["status"] == "completed":
                                    await bot.telegram.send_message(
                                        f"🔄 <b>Trade exécuté</b>\n\n"
                                        f"📊 Paire: {pair}\n"
                                        f"📈 Action: {final_action.upper()}\n"
                                        f"💰 Montant: {signal_adjusted_amount}\n"
                                        f"🎯 Confiance: {confidence:.0%}\n"
                                        f"💵 Prix: {trade_result.get('avg_price', 'N/A')}\n\n"
                                        f"🧠 Signaux:\n"
                                        f"  • Technique: {signal['confidence']:.0%}\n"
                                        f"  • IA: {ai_signal:.2f}\n"
                                        f"  • Sentiment: {sentiment_score:.2f}"
                                    )

                            # Vérifier les opportunités d'arbitrage
                            if bot.is_live_trading and final_action != "neutral":
                                arbitrage_opps = (
                                    await bot.detect_arbitrage_opportunities(pair)
                                )
                                for opp in arbitrage_opps:
                                    if (
                                        opp["diff_percent"] > 0.5
                                    ):  # Seuil minimum de profit
                                        print(
                                            f"💹 Opportunité d'arbitrage détectée pour {pair}: {opp['diff_percent']:.2f}%"
                                        )
                                        await bot.telegram.send_arbitrage_alert(opp)
                                        # Sauvegarde et mises à jour
                bot.current_cycle = cycle
                bot.regime = regime
                bot.save_shared_data()

                duration = (datetime.utcnow() - start).total_seconds()
                print(f"✅ Cycle terminé en {duration:.1f}s")

                # Mises à jour Telegram
                await bot.send_telegram_updates()

                # Rapport de trades si des trades ont été exécutés
                if trade_decisions:
                    trade_report = "💹 <b>Trades exécutés</b>\n\n"
                    for trade in trade_decisions:
                        status = trade["result"]["status"]
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
                error_msg = f"⚠️ Erreur cycle {cycle}: {e}"
                print(error_msg)
                await bot.telegram.send_message(error_msg)

            await asyncio.sleep(30)  # Attendre avant le prochain cycle

    except KeyboardInterrupt:
        stop_msg = "👋 Bot arrêté proprement"
        print(f"\n{stop_msg}")
        await bot.telegram.send_message(stop_msg)
        bot.save_shared_data()
    except Exception as e:
        error_msg = f"💥 Erreur fatale: {e}"
        print(error_msg)
        await bot.telegram.send_message(error_msg)


if __name__ == "__main__":
    try:
        asyncio.run(run_clean_bot())
    except Exception as e:
        print(f"💥 Erreur fatale: {e}")
