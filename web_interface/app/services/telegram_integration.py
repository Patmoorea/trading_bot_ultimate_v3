from telegram.ext import Application, CommandHandler, MessageHandler, filters
import telegram
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from ..config import Config

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bot = None
        self.app = None
        
        if not Config.TELEGRAM_ENABLED:
            self.logger.info("Telegram integration is disabled")
            return
            
        try:
            if not Config.TELEGRAM_TOKEN:
                raise ValueError("Telegram token not found in configuration")
                
            self.bot = telegram.Bot(token=Config.TELEGRAM_TOKEN)
            self.app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
            self.setup_handlers()
            self.chat_id = Config.TELEGRAM_CHAT_ID
            
            self.logger.info("Telegram bot initialized successfully")
        except Exception as e:
            self.logger.warning(f"Telegram integration disabled: {e}")
            self.bot = None
            self.app = None

    # ... reste du code inchangé ...

    async def send_alert(self, message: str, importance: str = "normal"):
        """Envoie une alerte via Telegram"""
        if not Config.TELEGRAM_ENABLED or not self.bot:
            self.logger.debug(f"Alert not sent (Telegram disabled): {message}")
            return
            
        try:
            if self.bot and self.chat_id:
                emoji = "🔴" if importance == "high" else "ℹ️"
                formatted_message = f"{emoji} {message}"
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_message,
                    parse_mode=telegram.constants.ParseMode.MARKDOWN
                )
            else:
                self.logger.debug("Telegram message not sent (bot not configured)")
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")

