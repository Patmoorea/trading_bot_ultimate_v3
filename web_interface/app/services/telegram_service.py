import asyncio
import telegram
from ..config import Config
import logging

class TelegramService:
    def __init__(self):
        self.bot = telegram.Bot(token=Config.TELEGRAM_TOKEN)
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.logger = logging.getLogger(__name__)

    async def send_alert(self, message, importance="normal"):
        try:
            emoji = "🚨" if importance == "high" else "ℹ️"
            formatted_message = f"{emoji} {message}"
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode=telegram.ParseMode.HTML
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
            return False

    async def send_trade_alert(self, trade_info):
        message = f"""
🤖 <b>Trade Signal</b>

Symbol: {trade_info['symbol']}
Action: {trade_info['action']}
Price: {trade_info['price']}
Confidence: {trade_info['confidence']:.2%}

Reason: {trade_info['reason']}
Timeframe: {trade_info['timeframe']}

Risk Level: {trade_info['risk_level']}
Stop Loss: {trade_info['stop_loss']}
        """
        await self.send_alert(message, importance="high")
