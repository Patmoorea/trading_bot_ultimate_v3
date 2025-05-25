from modules.trading.signal_processor import SignalProcessor
from modules.risk.advanced_risk import AdvancedRiskManager
from modules.news.sentiment_processor import NewsProcessor
from modules.utils.telegram_logger import TelegramLogger
import asyncio
from modules.arbitrage_engine import ArbitrageEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def safe_arbitrage():
    try:
        engine = ArbitrageEngine()
        opportunities = await asyncio.wait_for(
            asyncio.to_thread(engine.check_opportunities_v2),
            timeout=10.0
        )
        if opportunities:
            logger.info(f"Opportunités trouvées: {len(opportunities)}")
            for opp in opportunities:
                logger.info(f"{opp['pair']} - Spread: {opp['spread']*100:.2f}%")
        else:
            logger.info("Aucune opportunité arbitrage")
    except Exception as e:
        logger.error(f"Erreur arbitrage: {e}")

if __name__ == "__main__":
    import sys
    mode = "prod" if "--mode=prod" in sys.argv else "test"
    
    if mode == "prod":
        while True:
            asyncio.run(safe_arbitrage())
            asyncio.sleep(60)  # Toutes les 60 secondes
    else:
        asyncio.run(safe_arbitrage())

# Initialisation des nouveaux modules
news_analyzer = NewsProcessor()
tg_logger = TelegramLogger(config.TELEGRAM_TOKEN)

from modules.utils.advanced_logger import AdvancedLogger
global_logger = AdvancedLogger()

def handle_signal(signal):
    global_logger.log(f"Signal reçu: {signal}", notify=True)

from modules.utils.advanced_logger import AdvancedLogger
from datetime import datetime

global_logger = AdvancedLogger()

def handle_signal(signal):
    """Gère un signal de trading"""
    global_logger.log(f"Signal reçu: {signal}", notify=True)

# Initialisation des nouveaux modules
from modules.trading.signal_processor import SignalProcessor
from modules.risk.advanced_risk import AdvancedRiskManager

signal_processor = SignalProcessor()
risk_manager = AdvancedRiskManager()

def process_market_data(data):
    """Exemple d'intégration"""
    if data['bullish']:
        signal = signal_processor.add_signal(
            data['pair'], 
            'BUY', 
            data['confidence']
        )
        if data['confirmed']:
            signal_processor.confirm_signal(signal)
# Importation du modèle Keras
from model_keras import create_keras_model

# Utilisation du modèle
model = create_keras_model()
model.summary()

# Ajout en tête de fichier (après les autres imports)
from src.core_merged.performance import check_performance_threshold
from src.core_merged.monitoring import get_gpu_temp

def init_hardware():
    """Initialisation matérielle"""
    if not check_performance_threshold():
        raise RuntimeError('Vérification matérielle échouée')
    print('✅ Matériel validé')

def init_news_system():
    from src.analysis.sentiment.news_sentiment import NewsSentimentAnalyzer
    news_analyzer = NewsSentimentAnalyzer()
    return news_analyzer

def init_news_system():
    """
    Initialise le système d'analyse d'actualités
    Retourne: (NewsSentimentAnalyzer, bool) - (instance, status_ok)
    """
    try:
        from src.analysis.sentiment.news_sentiment import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        return analyzer, analyzer.sentiment_pipeline is not None
    except Exception as e:
        print(f"ERREUR initialisation news: {str(e)}")
        return None, False

if __name__ == "__main__":
    news_analyzer, status = init_news_system()
    if status:
        print("Système news opérationnel")
    else:
        print("Système news en mode dégradé")
