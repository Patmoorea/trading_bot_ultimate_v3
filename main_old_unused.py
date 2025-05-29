import sys
import os
import asyncio
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ajout chemins supplémentaires si besoin
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Imports corrigés avec 'src.' au début des chemins
from src.modules.trading.signal_processor import SignalProcessor
from src.modules.risk.advanced_risk import AdvancedRiskManager
from src.modules.news.sentiment_processor import NewsProcessor
from src.modules.utils.telegram_logger import TelegramLogger
from src.modules.utils.advanced_logger import AdvancedLogger
from src.src.strategies.arbitrage.multi_exchange.core.arbitrage_engine import MultiExchangeArbitrage as ArbitrageEngine

from src.model_keras import create_keras_model
from src.core.performance import check_performance_threshold
from src.core.monitoring import get_gpu_temp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

signal_processor = SignalProcessor()
risk_manager = AdvancedRiskManager()
global_logger = AdvancedLogger()

def init_hardware():
    logger.info("🔧 Vérification du matériel...")
    gpu_temp = get_gpu_temp()
    logger.info(f"Température GPU détectée : {gpu_temp} °C")
    if not check_performance_threshold():
        logger.error("❌ Échec de la vérification matérielle - Arrêt.")
        raise RuntimeError("Échec de la vérification matérielle")
    logger.info("✅ Matériel validé et performant.")

def init_news_system():
    try:
        from src.analysis.sentiment.news_sentiment import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        if analyzer.sentiment_pipeline is None:
            raise ValueError("Pipeline de sentiment non initialisée")
        logger.info("🗞️ Système d'analyse des news initialisé avec succès.")
        return analyzer, True
    except Exception as e:
        logger.warning(f"⚠️ Impossible d'initialiser l'analyse des news : {e}")
        return None, False

def load_config_and_logger():
    try:
        import config
        assert hasattr(config, "TELEGRAM_TOKEN"), "Token Telegram absent dans config.py"
        tg_logger = TelegramLogger(config.TELEGRAM_TOKEN)
        logger.info("✅ Logger Telegram initialisé.")
        return tg_logger
    except Exception as e:
        logger.warning(f"⚠️ Logger Telegram non initialisé : {e}")
        return None

def handle_signal(signal):
    msg = f"📡 Signal reçu : {signal}"
    global_logger.log(msg, notify=True)
    logger.info(msg)

def process_market_data(data):
    if data.get("bullish"):
        signal = signal_processor.add_signal(
            pair=data["pair"],
            direction="BUY",
            confidence=data["confidence"]
        )
        if data.get("confirmed"):
            signal_processor.confirm_signal(signal)

async def safe_arbitrage():
    try:
        engine = ArbitrageEngine()
        opportunities = await asyncio.wait_for(
            asyncio.to_thread(engine.check_opportunities_v2),
            timeout=15.0
        )
        if opportunities:
            logger.info(f"🟢 {len(opportunities)} opportunités d'arbitrage détectées")
            for opp in opportunities:
                logger.info(f"↔️ {opp['pair']} | Spread: {opp['spread'] * 100:.2f}%")
        else:
            logger.info("🔍 Aucune opportunité d'arbitrage détectée")
    except asyncio.TimeoutError:
        logger.warning("⏱️ Timeout lors de la détection d'arbitrage")
    except Exception as e:
        logger.error(f"❌ Erreur durant l'arbitrage : {e}")

async def main_loop_prod():
    logger.info("🌀 Démarrage boucle principale (mode PROD)")
    while True:
        await safe_arbitrage()
        await asyncio.sleep(60)

def main():
    print("🚀 Initialisation du système complet...")

    try:
        init_hardware()
    except Exception as e:
        print(f"❌ Erreur matériel: {e}")
        return

    model = create_keras_model()
    if "--summary" in sys.argv:
        model.summary()

    news_analyzer, news_ok = init_news_system()
    if news_ok:
        print("🗞️ Analyse des news opérationnelle")
    else:
        print("⚠️ Analyse des news désactivée")

    tg_logger = load_config_and_logger()

    mode = "prod" if "--mode=prod" in sys.argv else "test"
    logger.info(f"🔧 Mode actif : {mode.upper()}")

    if mode == "prod":
        asyncio.run(main_loop_prod())
    else:
        asyncio.run(safe_arbitrage())

if __name__ == "__main__":
    main()
