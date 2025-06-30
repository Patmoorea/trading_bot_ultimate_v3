import streamlit as st
import os
from dotenv import load_dotenv

import sys
import logging
import json
import re
import time
from datetime import timedelta
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import AsyncExitStack
import asyncio
import contextlib
import traceback

from src.bot.utils import StreamlitSessionManager
from src.bot.utils import _setup_and_verify_event_loop
from src.bot.streamlit_helpers import get_bot
from src.bot.ws import cleanup_resources
from src.bot.core import TradingBotM4
from src.notifications.telegram_bot import TelegramBot
from src.backtesting.advanced.quantum_backtest import QuantumBacktester, BacktestConfig
from src.backtesting.core.backtest_engine import BacktestEngine

# --- Chargement des variables d'environnement
load_dotenv()
logger = logging.getLogger(__name__)

print("BINANCE_TESTNET (os.environ):", os.environ.get("BINANCE_TESTNET"))
print("BINANCE_TESTNET (os.getenv):", os.getenv("BINANCE_TESTNET"))

USE_TESTNET = str(os.getenv("BINANCE_TESTNET", "False")).lower() in ("true", "1")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

st.set_page_config(
    page_title="Trading Bot Ultimate v4 - Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_FILE = "bot_status.json"


def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception as e:
                st.error(f"Erreur de lecture du status : {e}")
                return {}
    return {}


st.title("Trading Bot Ultimate v4 - Dashboard")

status = load_status()
if not status:
    st.warning(
        "Aucun status du bot trouvé. Le bot tourne-t-il ? (python bot_runner.py)"
    )
elif "error" in status:
    st.error(f"[BOT ERROR] {status['error']}")
else:
    st.success(f"Cycle : {status.get('cycle', '?')}")
    st.markdown(f"**Régime détecté :** {status.get('regime', '?')}")
    st.markdown(f"**Stratégie actuelle :** {status.get('strategy', '?')}")
    st.markdown(f"**Date/Heure :** {status.get('datetime', '?')}")
    st.markdown("**Signaux :**")
    st.json(status.get("signals", {}))

st.divider()
st.info(
    "Ce dashboard ne pilote pas le bot : il affiche uniquement le status en temps réel généré par le process autonome.\n\n"
    "Pour démarrer le bot : `python bot_runner.py`\n"
    "Pour surveiller, rafraîchez cette page."
)
if st.button("🔄 Rafraîchir le status"):
    st.rerun()


# --- INITIALISATION DES FLAGS ET DU BOT ---
async def ensure_bot_initialized(bot):
    # Initialise tous les composants asynchrones du bot (analyseurs, modèles, etc.)
    try:
        await bot._setup_components()
    except Exception as e:
        st.error(f"Erreur d'initialisation du bot TradingBotM4 : {e}")
        raise


if "bot" not in st.session_state or st.session_state["bot"] is None:
    bot = get_bot()
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(ensure_bot_initialized(bot))
    except Exception as e:
        st.error(f"Impossible d'initialiser le bot (analyseurs non prêts) : {e}")
        bot = None
    st.session_state["bot"] = bot
bot = st.session_state["bot"]

if "trading_task" not in st.session_state:
    st.session_state["trading_task"] = None
if "should_launch_bot" not in st.session_state:
    st.session_state["should_launch_bot"] = False
if "bot_running" not in st.session_state:
    st.session_state["bot_running"] = False

# --- CONTROLES UI START/STOP BOT ---
with st.sidebar:
    st.header("🛠️ Bot Controls")
    if not st.session_state.get("bot_running", False):
        if st.button("🟢 Start Trading", key="start_button", use_container_width=True):
            st.session_state["should_launch_bot"] = True
            st.rerun()
        st.success("Cliquez pour démarrer le bot.")
    else:
        if st.button("🔴 Stop Trading", key="stop_button", use_container_width=True):
            st.session_state["bot_running"] = False
            if st.session_state.get("trading_task"):
                st.session_state["trading_task"].cancel()
                st.session_state["trading_task"] = None
            st.warning("Trading stoppé.")

# --- LANCEMENT DE LA TÂCHE RUN_ADAPTIVE_TRADING ---
if st.session_state.get("should_launch_bot", False):
    st.session_state["bot_running"] = True
    if not st.session_state.get("trading_task"):
        loop = asyncio.get_event_loop()
        st.session_state["trading_task"] = loop.create_task(
            bot.run_adaptive_trading(period="7d")
        )
    st.session_state["should_launch_bot"] = False


# ================================
# ==== RESTE DE TON DASHBOARD ====
# ================================
# Helper pour valider les données OHLCV
def _has_valid_ohlcv(data):
    try:
        ohlcv = data.get("ohlcv") if isinstance(data, dict) else None
        if ohlcv is None or not isinstance(ohlcv, list):
            return False
        if len(ohlcv) == 0:
            return False
        for row in ohlcv:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                return False
        return True
    except Exception:
        return False


# Récupère les données du bot (adapte si besoin)
latest_data = getattr(bot, "latest_data", {}) if bot is not None else {}

data_ready = (
    isinstance(latest_data, dict)
    and bool(latest_data)
    and any(_has_valid_ohlcv(data) for data in latest_data.values())
)

if data_ready:
    # --- BACKTEST CLASSIQUE ---
    if st.button("Lancer Backtest", key="backtest_all_btn"):
        results = {}
        st.info("Backtest en cours sur toutes les paires...")
        try:
            for symbol, data in latest_data.items():
                try:
                    if _has_valid_ohlcv(data):
                        import pandas as pd

                        columns = [
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ]
                        df = pd.DataFrame(data["ohlcv"], columns=columns)
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                        def strategy_func(df, **params):
                            return (df["close"] > df["close"].rolling(5).mean()).astype(
                                int
                            )

                        # Adapter l'import si nécessaire !
                        from src.backtesting.core.backtest_engine import BacktestEngine

                        engine = BacktestEngine(initial_capital=10000)
                        results[symbol] = engine.run_backtest(df, strategy_func)
                    else:
                        st.warning(f"Aucune donnée OHLCV exploitable pour {symbol}")
                except Exception as pair_exc:
                    st.warning(f"Erreur sur {symbol}: {pair_exc}")
            st.session_state["all_backtest_results"] = results
            st.success("Backtest terminé ✅")
        except Exception as batch_exc:
            st.error(f"Erreur lors du backtest: {batch_exc}")

    # --- RÉSULTATS BACKTEST CLASSIQUE ---
    if st.session_state.get("all_backtest_results"):
        st.markdown("**Résultats Backtest Classique :**")
        for symbol, res in st.session_state["all_backtest_results"].items():
            st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")

    # --- BACKTEST QUANTIQUE ---
    if st.session_state.get("all_backtest_results"):
        if st.button("Lancer Backtest Quantique", key="quantum_backtest_all_btn"):
            st.info("Backtest quantique en cours sur toutes les paires...")
            results = {}
            try:
                from src.backtesting.advanced.quantum_backtest import QuantumBacktester
                import pandas as pd

                for symbol, data in latest_data.items():
                    st.write(f"Test {symbol} ...")
                    try:
                        if _has_valid_ohlcv(data):
                            columns = [
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ]
                            if len(data["ohlcv"]) > 0 and isinstance(
                                data["ohlcv"][0], (list, tuple)
                            ):
                                df = pd.DataFrame(data["ohlcv"], columns=columns)
                                df["timestamp"] = pd.to_datetime(
                                    df["timestamp"], unit="ms"
                                )
                            else:
                                df = pd.DataFrame(data["ohlcv"])

                            def strategy_func(df, **params):
                                return (
                                    df["close"] > df["close"].rolling(5).mean()
                                ).astype(int)

                            engine = QuantumBacktester()
                            results[symbol] = engine.run_quantum_simulation(
                                df, strategy_func
                            )
                        else:
                            st.warning(f"Aucune donnée OHLCV exploitable pour {symbol}")
                    except Exception as pair_exc:
                        st.warning(f"Erreur quantique sur {symbol}: {pair_exc}")
                st.session_state["all_quantum_results"] = results
                st.success("Backtest quantique terminé ✅")
                st.write("DEBUG - Résultats quantum :", results)
            except Exception as batch_exc:
                st.error(f"Erreur lors du backtest quantique: {batch_exc}")

    # --- RÉSULTATS BACKTEST QUANTIQUE ---
    if st.session_state.get("all_quantum_results"):
        st.markdown("**Résultats Backtest Quantique :**")
        for symbol, res in st.session_state["all_quantum_results"].items():
            st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")


# Exemples de helpers (garde-les, adapte si besoin)
async def cancel_trading_task():
    trading_task = st.session_state.get("trading_task")
    if trading_task is not None:
        trading_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await trading_task
        st.session_state["trading_task"] = None


# Onglet Portfolio
async def _render_portfolio_tab(bot):
    if st.session_state.bot_running:
        try:
            portfolio = st.session_state.get("portfolio")
            if portfolio:
                import pandas as pd

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "💰 Total Value",
                        f"{portfolio.get('total_value', 0):.2f} USDC",
                        f"{portfolio.get('daily_pnl', 0):+.2f} USDC",
                    )
                with col2:
                    st.metric(
                        "📈 24h Volume",
                        f"{portfolio.get('volume_24h', 0):.2f} USDC",
                        f"{portfolio.get('volume_change', 0):+.2f}%",
                    )
                with col3:
                    positions = portfolio.get("positions", [])
                    st.metric(
                        "🔄 Active Positions",
                        str(len(positions)),
                        f"{len(positions)} active",
                    )
                if positions:
                    st.subheader("Active Positions")
                    st.dataframe(pd.DataFrame(positions), use_container_width=True)
                else:
                    st.info("💡 No active positions")
            else:
                st.warning("⚠️ Waiting for portfolio data...")
        except Exception as e:
            st.error(f"❌ Portfolio error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view portfolio")


# Onglet Trading
async def _render_trading_tab(bot):
    if st.session_state.bot_running:
        try:
            latest_data = bot.latest_data.get("BTCUSDT", {})
            if latest_data:
                import pandas as pd

                col1, col2 = st.columns(2)
                with col1:
                    current_price = latest_data[-1]["close"]
                    prev_price = (
                        latest_data[-2]["close"]
                        if len(latest_data) > 1
                        else current_price
                    )
                    price_change = (
                        ((current_price - prev_price) / prev_price * 100)
                        if prev_price
                        else 0
                    )
                    st.metric(
                        "BTC/USDC Price",
                        f"{current_price:.2f}",
                        f"{price_change:+.2f}%",
                    )
                with col2:
                    current_vol = latest_data[-1]["volume"]
                    prev_vol = (
                        latest_data[-2]["volume"]
                        if len(latest_data) > 1
                        else current_vol
                    )
                    vol_change = (
                        ((current_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
                    )
                    st.metric(
                        "Trading Volume", f"{current_vol:.2f}", f"{vol_change:+.2f}%"
                    )
            if bot.indicators:
                st.subheader("Trading Signals")
                st.dataframe(pd.DataFrame(bot.indicators), use_container_width=True)
            else:
                st.info("💡 Waiting for signals...")
        except Exception as e:
            st.error(f"❌ Trading data error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view signals")


# Onglet Analysis
async def _render_analysis_tab(bot):
    if st.session_state.bot_running:
        try:
            import pandas as pd

            if bot.latest_data and bot.indicators:
                st.subheader("Technical Analysis")
                for symbol in bot.latest_data:
                    await process_market_data(bot, symbol)
                if hasattr(bot, "advanced_indicators"):
                    analysis = bot.advanced_indicators.get_all_signals()
                    st.dataframe(pd.DataFrame(analysis), use_container_width=True)
                else:
                    st.info("💡 Processing analysis...")
            else:
                st.info("💡 Waiting for market data...")
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
    else:
        st.warning("⚠️ Start trading to view analysis")
    if hasattr(bot, "qsvm") and bot.qsvm is not None:
        try:
            features = bot.latest_data
            quantum_signal = bot.qsvm.predict(features)
            st.subheader("Quantum SVM Signal")
            st.metric("Quantum SVM Signal", quantum_signal)
        except Exception as e:
            st.warning(f"Erreur Quantum SVM : {e}")


# Ajout: Hack JavaScript pour autorefresh sans st_autorefresh
def auto_refresh(interval_ms=2000, key="js_autorefresh"):
    js_code = f"""
    <script>
        if (!window.{key}) {{
            window.{key} = setInterval(function() {{
                window.location.reload();
            }}, {interval_ms});
        }}
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


# Initialisation des flags de protection
for flag, default in [
    ("prevent_cleanup", True),
    ("keep_alive", True),
    ("force_cleanup", False),
    ("cleanup_allowed", False),
]:
    if flag not in st.session_state:
        st.session_state[flag] = default
