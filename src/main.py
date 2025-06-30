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
if "bot" not in st.session_state or st.session_state["bot"] is None:
    st.session_state["bot"] = get_bot()
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

# Onglets principaux (tu adaptes si besoin)
try:
    portfolio_tab, trading_tab, analysis_tab = st.tabs(
        ["📈 Portfolio", "🎯 Trading", "📊 Analysis"]
    )
    with portfolio_tab:
        st.write(
            "Portfolio Tab"
        )  # Tu ajoutes ici l'appel à ta fonction/logiciel métier
        # await _render_portfolio_tab(bot)  # si besoin, décommente
    with trading_tab:
        st.write("Trading Tab")
        # await _render_trading_tab(bot)
    with analysis_tab:
        st.write("Analysis Tab")
        # await _render_analysis_tab(bot)
except Exception as tab_error:
    logger.error(f"Tab rendering error: {tab_error}")
    st.error("Error rendering tabs")


# Fonctions auxiliaires pour le rendu des onglets
async def _render_portfolio_tab(bot):
    """Rendu de l'onglet Portfolio"""
    if st.session_state.bot_running:
        try:
            portfolio = st.session_state.get("portfolio")
            if portfolio:
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


async def _render_trading_tab(bot):
    """Rendu de l'onglet Trading"""
    if st.session_state.bot_running:
        try:
            latest_data = bot.latest_data.get("BTCUSDT", {})
            if latest_data:
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


async def _render_analysis_tab(bot):
    """Rendu de l'onglet Analysis"""
    if st.session_state.bot_running:
        try:
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

    # --- Signal Quantum SVM ---
    if hasattr(bot, "qsvm") and bot.qsvm is not None:
        try:
            # Prépare les features à passer à predict (adapte cette ligne selon ta logique)
            features = (
                bot.latest_data
            )  # ou bot.indicators ou ton dataframe, adapte selon besoin
            quantum_signal = bot.qsvm.predict(features)
            st.subheader("Quantum SVM Signal")
            st.metric("Quantum SVM Signal", quantum_signal)
        except Exception as e:
            st.warning(f"Erreur Quantum SVM : {e}")


# --- Ajout: Hack JavaScript pour autorefresh sans st_autorefresh ---
def auto_refresh(interval_ms=2000, key="js_autorefresh"):
    """Inject JS code for auto-refresh in Streamlit."""
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
