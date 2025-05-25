import os
import logging
from dotenv import load_dotenv
from core.technical_engine import TechnicalEngine
from core.risk_manager import RiskManager

# Chargement de la configuration
load_dotenv()
config = {
    "NEWS": {
        "enabled": True,
        "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", "")
    }
}

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    init_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("⚡ Initialisation du Trading Bot M4")
        engine = TechnicalEngine()
        risk_mgr = RiskManager()
        
        # Simulation de données
        test_data = [1.5, 2.3, 3.1, 4.0]
        
        logger.info("Analyse des données...")
        signal = engine.compute(test_data)
        risk_assessment = risk_mgr.evaluate_risk(signal)
        
        logger.info(f"Signal: {signal}")
        logger.info(f"Évaluation du risque: {risk_assessment}")
        
    except KeyboardInterrupt:
        logger.info("Arrêt manuel demandé")
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
    finally:
        logger.info("Bot arrêté")

if __name__ == "__main__":
    main()

# ========== NOUVEAUX MODULES ==========
from news_processor.core import NewsSentimentAnalyzer
from regime_detection.hmm_kmeans import MarketRegimeDetector
from liquidity_heatmap.visualization import generate_heatmap
from quantum_ml.qsvm import QuantumSVM

class TradingBot:
    def __init__(self):
        # [...] Code existant conservé
        self.news_analyzer = NewsSentimentAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.qsvm = QuantumSVM()

    def update_heatmap(self):
        orderbook = self.exchange.fetch_order_book("BTC/USDT")
        self.current_heatmap = generate_heatmap(orderbook)

# ========== IMPORT OPTIMISÉ (AJOUT SEULEMENT) ==========
try:
    from src.news_processor.core import CachedNewsSentimentAnalyzer as NewsAnalyzer
    from src.regime_detection.hmm_kmeans import OptimizedMarketRegimeDetector as RegimeDetector
    print("✓ Versions optimisées chargées")
except ImportError:
    from src.news_processor.core import NewsSentimentAnalyzer as NewsAnalyzer
    from src.regime_detection.hmm_kmeans import MarketRegimeDetector as RegiseDetector
    print("ℹ Versions standard chargées (fallback)")

# Utilisation transparente :
bot_analyzer = NewsAnalyzer()  # Utilise automatiquement la version optimisée si disponible
bot_regime_detector = RegimeDetector()
