import asyncio
import json
import os
from datetime import datetime
from src.bot.core import TradingBotM4
from src.notifications.telegram_bot import TelegramBot
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

STATUS_PATH = "bot_status.json"


def write_status(data):
    with open(STATUS_PATH, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    bot = TradingBotM4()
    await bot.initialize()
    print("[BOT] Bot initialisé, démarrage de la boucle adaptative...")

    try:
        cycle = 0
        while True:
            cycle += 1
            # --- 1. Exécution d'un cycle classique (ton code existant) ---
            regime, historical_data, indicators_analysis = await bot.study_market("7d")
            strategy = bot.choose_strategy(regime, indicators_analysis)
            market_data = await bot.get_latest_data()
            signals = await bot.analyze_signals(market_data)
            # --- 2. Ecriture du status pour Streamlit ---
            status = {
                "cycle": cycle,
                "regime": regime,
                "strategy": strategy,
                "signals": signals,
                "datetime": datetime.utcnow().isoformat(),
            }
            write_status(status)
            # --- 3. Sleep entre deux cycles ---
            await asyncio.sleep(5)
    except Exception as e:
        print("[BOT] Exception dans la boucle principale:", e)
        write_status({"error": str(e)})


if __name__ == "__main__":
    asyncio.run(main())
