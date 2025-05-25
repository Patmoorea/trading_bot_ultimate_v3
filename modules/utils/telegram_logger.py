import asyncio
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()


class TelegramLogger:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = aiohttp.ClientTimeout(total=2)
        self._session = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self):
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            return self._session

    async def _send_message(self, message):
        if not self.token or not self.chat_id:
            return False

        session = None
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            params = {"chat_id": self.chat_id, "text": message}
            async with session.post(url, params=params, timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"Erreur Telegram: {str(e)}")
            return False
        finally:
            # On ne ferme pas la session pour la réutiliser
            pass

    def log(self, message):
        try:
            return asyncio.run(self._send_message(message))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._send_message(message))
            loop.close()
            return result

    async def close(self):
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except BaseException:
            pass
