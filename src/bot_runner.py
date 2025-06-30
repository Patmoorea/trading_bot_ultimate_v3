import asyncio
import json
import os
import traceback
from datetime import datetime
from src.bot.core import TradingBotM4
from src.notifications.telegram_bot import TelegramBot

current_dir = os.path.dirname(os.path.abspath(__file__))

STATUS_PATH = "bot_status.json"


def write_status(data):
    with open(STATUS_PATH, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    bot = TradingBotM4()
    # --- Initialisation complète des analyseurs et modèles ---
    await bot._setup_components()  # <-- AJOUT CRITIQUE

    print("[BOT] Bot initialisé, démarrage de la boucle adaptative...")

    try:
        cycle = 0
        while True:
            cycle += 1
            try:
                # --- 1. Exécution d'un cycle classique ---
                regime, historical_data, indicators_analysis = await bot.study_market(
                    "7d"
                )
                strategy = bot.choose_strategy(regime, indicators_analysis)
                market_data = await bot.get_latest_data()
                signals = {}
                for pair in bot.pairs_valid:
                    pair_key = pair if pair in market_data else pair.replace("/", "")
                    if pair_key in market_data:
                        pair_data = market_data[pair_key]
                        signals[pair] = await bot.analyze_signals(pair_data)
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
            except Exception as loop_e:
                err_text = (
                    f"{type(loop_e).__name__}: {loop_e}\n{traceback.format_exc()}"
                )
                print("[BOT] Exception dans un cycle de la boucle:", err_text)
                write_status({"error": err_text})
                await asyncio.sleep(5)
    except Exception as e:
        err_text = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print("[BOT] Exception fatale:", err_text)
        write_status({"error": err_text})


if __name__ == "__main__":
    asyncio.run(main())
