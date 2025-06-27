# 1. Import et configuration Streamlit (DOIT ÊTRE EN PREMIER)
import streamlit as st
import os
# Charger les variables d'env
from dotenv import load_dotenv

# 2. Imports système
import sys
import logging
import json
import re
import time
import signal
from datetime import timedelta
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import AsyncExitStack
from asyncio import TimeoutError, AbstractEventLoop
import asyncio
import nest_asyncio
import aiohttp
import traceback

load_dotenv()

print("BINANCE_TESTNET (os.environ):", os.environ.get("BINANCE_TESTNET"))
print("BINANCE_TESTNET (os.getenv):", os.getenv("BINANCE_TESTNET"))

USE_TESTNET = str(os.getenv("BINANCE_TESTNET", "False")).lower() in ("true", "1")

# Configuration des chemins
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
def main():
    """Point d'entrée principal avec protection renforcée et gestion des événements améliorée"""
    current_time = datetime.now(timezone.utc)
    current_user = os.getenv("USER", "Patmoorea")

    try:
        # 1. Initialisation et protection de la session
        global session_manager
        session_manager = StreamlitSessionManager()
        session_manager.protect_session()

        # 2. Log de démarrage
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              STARTING APPLICATION                ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
║ Session: {session_manager.session_id}
╚═════════════════════════════════════════════════╝
        """
        )

        # 3. Initialisation de l'état de session
        _initialize_session_state()

        # 4. Configuration et vérification de la boucle d'événements
        event_loop = _setup_and_verify_event_loop()
        if not event_loop:
            raise RuntimeError("Failed to initialize event loop")

        # 5. Exécution de la coroutine principale
        event_loop.run_until_complete(main_async())

    except asyncio.CancelledError:
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              GRACEFUL SHUTDOWN                   ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {current_user}
╚═════════════════════════════════════════════════╝
        """
        )

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              RUNTIME ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
║ User: {current_user}
╚═════════════════════════════════════════════════╝
        """
        )
        st.error(f"❌ Application error: {str(e)}")

    finally:
        _perform_cleanup()


async def main_async():
    """Point d'entrée principal de l'application avec gestion améliorée des états"""

    # --- DEBUG state au tout début ---
    def debug_state(when):
        import streamlit as st

        st.write(
            f"DEBUG [{when}] bot_running={st.session_state.get('bot_running')}, should_launch_bot={st.session_state.get('should_launch_bot')}, trading_task={st.session_state.get('trading_task')}"
        )
        print(
            f"DEBUG [{when}] bot_running={st.session_state.get('bot_running')}, should_launch_bot={st.session_state.get('should_launch_bot')}, trading_task={st.session_state.get('trading_task')}"
        )

    debug_state("DEBUT")

    # ---- 1. Initialisation des flags et objets critiques AVANT tout le reste ----
    if "bot" not in st.session_state or st.session_state["bot"] is None:
        st.session_state["bot"] = get_bot()
    bot = st.session_state["bot"]

    if "trading_task" not in st.session_state:
        st.session_state["trading_task"] = None
    if "should_launch_bot" not in st.session_state:
        st.session_state["should_launch_bot"] = False

    # ---- 2. Lancement du bot trading si flag actif (AVANT tout reset de state) ----
    if st.session_state.get("should_launch_bot", False):
        st.session_state["bot_running"] = True
        st.session_state["should_launch_bot"] = False  # reset le flag
        if not st.session_state.get("trading_task"):
            loop = st.session_state.get("loop") or asyncio.get_event_loop()
            st.session_state["trading_task"] = loop.create_task(
                bot.run_adaptive_trading(period="7d")
            )
    debug_state("APRES_LA_GESTION_LAUNCH_BOT")

    # ---- 3. Initialisation du reste du state par défaut ----
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    current_user = "Patmoorea"

    try:
        session_manager.protect_session()

        if "initialization_time" not in st.session_state:
            st.session_state.initialization_time = current_time

        default_session_state = {
            "session_id": f"{current_user}_{int(datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S').timestamp())}",
            "user": current_user,
            "portfolio": None,
            "latest_data": None,
            "indicators": None,
            # ATTENTION : on ne reset PAS bot_running ici pour ne pas casser le flag !
            "refresh_count": 0,
            "ws_status": "disconnected",
            "ws_initialized": False,
            "ws_connection_status": "disconnected",
            "last_update_time": current_time,
            "needs_update": False,
            "update_interval": 2.0,
            "last_refresh": time.time(),
        }
        for key, value in default_session_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

        if bot is None:
            st.error("❌ Failed to initialize bot")
            return

        await bot._initialize_analyzers()

        status_placeholder = st.sidebar.empty()

        # --- DEBUG données disponibles ---
        st.sidebar.markdown("#### Données présentes dans bot.latest_data :")

        latest_data = st.session_state.get("latest_data", {})
        if not isinstance(latest_data, dict):
            latest_data = {}
        st.sidebar.write(
            {k: getattr(v, "shape", str(type(v))) for k, v in latest_data.items()}
            if latest_data
            else "Aucune donnée"
        )

        # 5. Interface principale - État et contrôles
        status_col1, status_col2 = st.columns([2, 1])
        with status_col1:
            ws_status = st.session_state.get("ws_connection_status", "disconnected")
            ws_icon = {
                "connected": "🟢",
                "disconnected": "🔴",
                "initializing": "🔄",
                "error": "⚠️",
            }.get(ws_status, "🔴")

            status_info = f"""
            ### Bot Status
            - 🚦 Trading: {'🟢 Active' if st.session_state.get('bot_running') else '🔴 Stopped'}
            - 📡 WebSocket: {ws_icon} {ws_status.title()}
            - 💼 Portfolio: {'✅ Available' if st.session_state.portfolio else '⚠️ Not Available'}
            - ⏰ Last Update: {st.session_state.last_update_time}
            - 👤 User: {st.session_state.user}
            """
            st.info(status_info)

        # 6. Contrôles de la barre latérale avec gestion améliorée
        with st.sidebar:
            st.header("🛠️ Bot Controls")
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Low",
                key=f"risk_level_slider_{st.session_state.session_id}",
            )
            st.divider()

            # --- CONTROLES BOT TRADING ---
            if not st.session_state.get("bot_running", False):
                if st.button(
                    "🟢 Start Trading", key="start_button", use_container_width=True
                ):
                    st.session_state["should_launch_bot"] = True
                    st.rerun()  # ←←← AJOUT OBLIGATOIRE
                st.success("Cliquez pour démarrer le bot.")
            else:
                if st.button(
                    "🔴 Stop Trading", key="stop_button", use_container_width=True
                ):
                    st.session_state["bot_running"] = False
                    if st.session_state.get("trading_task"):
                        st.session_state["trading_task"].cancel()
                        st.session_state["trading_task"] = None
                    st.warning("Trading stoppé.")

            debug_state("APRES_BOUTONS")

            # --- AFFICHAGE LIVE ---
            if "live_status" in st.session_state and st.session_state["live_status"]:
                status_placeholder.markdown("### 🟢 Trading Live Status")
                for k, v in st.session_state["live_status"].items():
                    status_placeholder.write(f"**{k}** : {v}")

            # --- GESTION DES DONNEES ET BACKTEST ---
            latest_data = st.session_state.get("latest_data")
            if not isinstance(latest_data, dict):
                latest_data = {}
            st.write("DEBUG - latest_data:", latest_data)  # <-- À enlever ensuite

            def _has_valid_ohlcv(item):
                return (
                    isinstance(item, dict)
                    and "ohlcv" in item
                    and isinstance(item["ohlcv"], list)
                    and len(item["ohlcv"]) > 0
                    and isinstance(item["ohlcv"][0], dict)
                    and all(
                        k in item["ohlcv"][0]
                        for k in ["timestamp", "open", "high", "low", "close", "volume"]
                    )
                )

            data_ready = any(_has_valid_ohlcv(item) for item in latest_data.values())

            if not data_ready:
                st.warning(
                    "Aucune donnée OHLCV disponible. Clique sur le bouton ci-dessous pour charger les données de marché."
                )
                if st.button("Charger les données", key="load_data_btn"):
                    with st.spinner("Chargement des données..."):
                        loaded = False
                        try:
                            if not hasattr(bot, "binance_ws") or bot.binance_ws is None:
                                st.info("Initialisation de la WebSocket…")
                                await bot.initialize()
                            if hasattr(bot, "get_latest_data"):
                                data = await bot.get_latest_data()
                                st.write("DEBUG - Résultat get_latest_data:", data)
                                if data and isinstance(data, dict) and len(data) > 0:
                                    st.session_state["latest_data"] = data
                                    loaded = True
                                else:
                                    st.error(
                                        "La récupération a retourné None ou un dict vide : pas de données."
                                    )
                            elif hasattr(bot, "load_all_data"):
                                await bot.load_all_data()
                                latest_data = getattr(bot, "latest_data", {}) or {}
                                if not isinstance(latest_data, dict):
                                    latest_data = {}
                                st.write(
                                    "DEBUG - latest_data après load_all_data:",
                                    latest_data,
                                )
                                loaded = (
                                    isinstance(latest_data, dict)
                                    and len(latest_data) > 0
                                )
                                if loaded:
                                    st.session_state["latest_data"] = latest_data
                                else:
                                    st.error(
                                        "La récupération a retourné None ou un dict vide : pas de données."
                                    )
                            else:
                                st.error(
                                    "Aucune méthode de chargement trouvée sur le bot."
                                )
                        except Exception as exc:
                            st.error(f"Erreur lors du chargement des données : {exc}")
                        if loaded:
                            st.success("Données chargées ! Tu peux lancer un backtest.")
                            st.rerun()
            else:
                # --- BACKTEST CLASSIQUE ---
                if st.button("Lancer Backtest", key="backtest_all_btn"):
                    results = {}
                    st.info("Backtest en cours sur toutes les paires...")
                    try:
                        for symbol, data in latest_data.items():
                            try:
                                if _has_valid_ohlcv(data):
                                    import pandas as pd

                                    df = pd.DataFrame(data["ohlcv"])

                                    def strategy_func(df, **params):
                                        return (
                                            df["close"] > df["close"].rolling(5).mean()
                                        ).astype(int)

                                    engine = BacktestEngine(initial_capital=10000)
                                    results[symbol] = engine.run_backtest(
                                        df, strategy_func
                                    )
                                else:
                                    st.warning(
                                        f"Aucune donnée OHLCV exploitable pour {symbol}"
                                    )
                            except Exception as pair_exc:
                                st.warning(f"Erreur sur {symbol}: {pair_exc}")
                        st.session_state["all_backtest_results"] = results
                        st.success("Backtest terminé ✅")
                    except Exception as batch_exc:
                        st.error(f"Erreur lors du backtest: {batch_exc}")

                # Résultats
                if st.session_state.get("all_backtest_results"):
                    st.markdown("**Résultats Backtest Classique :**")
                    for symbol, res in st.session_state["all_backtest_results"].items():
                        st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")

                    if st.button(
                        "Lancer Backtest Quantique", key="quantum_backtest_all_btn"
                    ):
                        st.info("Backtest quantique en cours sur toutes les paires...")
                        results = {}
                        if not isinstance(latest_data, dict):
                            latest_data = {}
                        st.write(
                            "DEBUG - Paire/Data dispo :",
                            {
                                k: getattr(v, "shape", str(type(v)))
                                for k, v in latest_data.items()
                            },
                        )
                        try:
                            for symbol, data in latest_data.items():
                                st.write(f"Test {symbol} ...")
                                try:
                                    if _has_valid_ohlcv(data):
                                        import pandas as pd

                                        df = pd.DataFrame(data["ohlcv"])

                                        def strategy_func(df, **params):
                                            return (
                                                df["close"]
                                                > df["close"].rolling(5).mean()
                                            ).astype(int)

                                        engine = BacktestEngine(initial_capital=10000)
                                        results[symbol] = engine.run_backtest(
                                            df, strategy_func
                                        )
                                    else:
                                        st.warning(
                                            f"Aucune donnée OHLCV exploitable pour {symbol}"
                                        )
                                except Exception as pair_exc:
                                    st.warning(
                                        f"Erreur quantique sur {symbol}: {pair_exc}"
                                    )
                            st.session_state["all_quantum_results"] = results
                            st.success("Backtest quantique terminé ✅")
                            st.write("DEBUG - Résultats quantum :", results)
                        except Exception as batch_exc:
                            st.error(f"Erreur lors du backtest quantique: {batch_exc}")

                if st.session_state.get("all_quantum_results"):
                    st.markdown("**Résultats Backtest Quantique :**")
                    for symbol, res in st.session_state["all_quantum_results"].items():
                        st.write(f"{symbol} : {res.get('final_capital', 'N/A')} USD")

        # 8. Onglets principaux avec gestion d'erreur
        try:
            portfolio_tab, trading_tab, analysis_tab = st.tabs(
                ["📈 Portfolio", "🎯 Trading", "📊 Analysis"]
            )

            # Onglet Portfolio
            with portfolio_tab:
                await _render_portfolio_tab(bot)

            # Onglet Trading
            with trading_tab:
                await _render_trading_tab(bot)

            # Onglet Analysis
            with analysis_tab:
                await _render_analysis_tab(bot)
        except Exception as tab_error:
            logger.error(f"Tab rendering error: {tab_error}")
            st.error("Error rendering tabs")

    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        st.error(f"❌ Application error: {str(e)}")
    finally:
        # Protection finale avec timestamp
        try:
            session_manager.protect_session()
            st.session_state.last_update_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception as protect_error:
            logger.error(f"Session protection error: {protect_error}")

    debug_state("FIN")


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
  
   st.set_page_config(
    page_title="Trading Bot Ultimate v4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
            
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
        
if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║              KEYBOARD INTERRUPT                  ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Status: Graceful shutdown initiated
╚═════════════════════════════════════════════════╝
        """
        )

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║              CRITICAL ERROR                      ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        sys.exit(1)

    finally:
        try:
            # Nettoyage final avec nouvelle boucle si nécessaire
            if "bot_instance" in st.session_state:
                try:
                    # Création d'une nouvelle boucle pour le nettoyage final
                    cleanup_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(cleanup_loop)
                    cleanup_loop.run_until_complete(
                        cleanup_resources(st.session_state.bot_instance)
                    )
                    cleanup_loop.close()
                except Exception as e:
                    logger.error(f"Final cleanup error: {e}")

            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║              FINAL CLEANUP                       ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Status: All resources cleaned
╚═════════════════════════════════════════════════╝
            """
            )

        except Exception as cleanup_error:
            logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║              CLEANUP ERROR                       ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Error: {str(cleanup_error)}
╚═════════════════════════════════════════════════╝
            """
            )
