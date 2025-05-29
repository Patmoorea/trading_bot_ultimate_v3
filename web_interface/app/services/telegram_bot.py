import logging
from app.config import Config

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enabled = Config.TELEGRAM_ENABLED
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        
    async def send_message(self, message):
        if not self.enabled:
            return
            
        try:
            # Logique d'envoi de message
            self.logger.info(f"Telegram message sent: {message}")
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
