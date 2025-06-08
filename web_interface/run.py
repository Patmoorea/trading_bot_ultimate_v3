import logging
from datetime import datetime, timezone
from app import create_app, socketio
from app.config import Config
from app.services.trading_engine import TradingEngine
from app.services.websocket_manager import WebSocketManager
from app.services.telegram_bot import TelegramBot

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    # Current time in UTC
    current_time = datetime.now(timezone.utc)

    # Banner d'information
    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║                 Trading Bot Ultimate Interface                ║
╠══════════════════════════════════════════════════════════════╣
║ Time: %s UTC                                    ║
║ User: %s                                             ║
║ Mode: %s                                         ║
║ Telegram: %s                                        ║
║ Server: http://localhost:%d                               ║
╚══════════════════════════════════════════════════════════════╝
    """, current_time.strftime("%Y-%m-%d %H:%M:%S"),
         Config.CURRENT_USER,
         Config.TRADING_MODE,
         "Enabled" if Config.TELEGRAM_ENABLED else "Disabled",
         Config.PORT)

    # Création de l'application
    app = create_app()
    
    # Initialisation des composants
    trading_engine = TradingEngine()
    websocket_manager = WebSocketManager()
    telegram_bot = TelegramBot() if Config.TELEGRAM_ENABLED else None

    logger.info("Starting Trading Bot Ultimate...")
    logger.info("Server will be available at http://localhost:%d", Config.PORT)
    logger.info("Press Ctrl+C to stop the server")

    # Démarrage du serveur
    socketio.run(app, 
                 host=Config.HOST,
                 port=Config.PORT,
                 debug=Config.DEBUG)

if __name__ == "__main__":
    main()
