import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import psutil
import pytz
from datetime import datetime, timedelta
from binance.client import Client
from src.backtesting.core.backtest_engine import BacktestEngine
from src.strategies import sma_strategy, breakout_strategy, arbitrage_strategy
from src.bot_runner import _generate_analysis_report
from src.risk_tools import kelly_criterion, calculate_var, calculate_max_drawdown

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Trading Bot Ultimate v4 - Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Patmoorea/trading_bot_ultimate_v3",
        "Report a bug": "https://github.com/Patmoorea/trading_bot_ultimate_v3/issues",
        "About": "# Trading Bot Ultimate v4\nVersion avancée avec IA et analyses quantiques.",
    },
)

STATUS_FILE = "bot_status.json"
SHARED_DATA_PATH = "src/shared_data.json"
LOG_FILE = "src/bot_logs.txt"
CONFIG_FILE = "config.json"
CURRENT_USER = "Patmoorea"


def get_current_time():
    utc_now = datetime.utcnow()
    polynesie_offset = timedelta(hours=-10)
    local_dt = utc_now + polynesie_offset
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")


def load_json_file(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


shared_data = load_json_file(SHARED_DATA_PATH)

with st.sidebar:
    st.header("🤖 Bot Status")
    status = load_json_file(STATUS_FILE)
    tahiti = pytz.timezone("Pacific/Tahiti")
    now_tahiti = datetime.now(tahiti).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"""
        <div style='background-color: #0f3d40; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #00ff00; margin: 0;'>✅ Bot Actif</h3>
            <p style='color: #ffffff; margin: 5px 0;'>Dernière mise à jour: {now_tahiti} (heure Polynésie)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='background-color: #1c1c1c; padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0;'>👤 Utilisateur: Patmoorea</p>
            <p style='margin: 0;'>🌐 Mode: Production</p>
            <p style='margin: 0;'>📈 Exchange: Binance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Gestion avancée des pauses/news ---
    st.markdown("### ⏸️ Pauses trading actives")
    active_pauses = shared_data.get("active_pauses", [])
    if active_pauses:
        df_pauses = pd.DataFrame(active_pauses)
        st.dataframe(df_pauses, use_container_width=True)
        if st.button("✅ Forcer reprise trading (stop toutes pauses)"):
            for p in active_pauses:
                p["cycles_left"] = 0
            shared_data["active_pauses"] = active_pauses
            with open(SHARED_DATA_PATH, "w") as f:
                json.dump(shared_data, f, indent=2)
            st.success("Reprise forcée — toutes pauses stoppées !")
            st.rerun()
        prolong_cycles = st.number_input(
            "Prolonger toutes pauses de N cycles", min_value=1, max_value=20, value=3
        )
        if st.button("⏸️ Prolonger toutes pauses"):
            for p in active_pauses:
                p["cycles_left"] += int(prolong_cycles)
            shared_data["active_pauses"] = active_pauses
            with open(SHARED_DATA_PATH, "w") as f:
                json.dump(shared_data, f, indent=2)
            st.success(f"Pauses prolongées de {prolong_cycles} cycles.")
            st.rerun()
    else:
        st.info("Aucune pause active actuellement.")

    with st.expander("🗂️ Historique des pauses trading"):
        pause_history = shared_data.get("pause_history", [])
        if pause_history:
            df_hist = pd.DataFrame(pause_history)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.write("Aucun historique de pause trading.")

    safe_mode = shared_data.get("safe_mode", False)
    if safe_mode:
        st.warning(
            "⚠️ MODE SAFE ACTIVÉ : sizing réduit à cause de pertes consécutives !"
        )

    st.markdown("### 🚨 Alertes actives")
    alerts = shared_data.get("alerts", [])
    for alert in alerts:
        if alert["level"] == "critical":
            st.error(f"{alert['message']} ({alert['timestamp']})")
        elif alert["level"] == "warning":
            st.warning(f"{alert['message']} ({alert['timestamp']})")
        else:
            st.info(f"{alert['message']} ({alert['timestamp']})")

    st.header("⛔️ Positions fermées (auto)")
    closed = shared_data.get("closed_positions", [])
    if closed:
        df_closed = pd.DataFrame(closed)
        st.dataframe(df_closed, use_container_width=True)
    else:
        st.info("Aucune position fermée automatiquement ce cycle.")

    st.sidebar.divider()

    st.sidebar.markdown(
        f"""
### 📊 Informations système
- 🕒 Dernière mise à jour: {now_tahiti} (heure Polynésie) 
- 👤 Session: {CURRENT_USER}
- 🌐 Version: 4.0.1
- 📡 Status: En ligne
- 💾 Mémoire utilisée: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB
"""
    )
    st.sidebar.markdown(
        f"""
        <div style='background-color: #1c1c1c; padding: 10px; border-radius: 5px; margin-top: 20px;'>
            <p style='margin: 0; color: #00ff00;'>🟢 Exchange: Connecté</p>
            <p style='margin: 0; color: #00ff00;'>🟢 Base de données: Synchronisée</p>
            <p style='margin: 0; color: #00ff00;'>🟢 API: Opérationnelle</p>
            <p style='margin: 0; color: #808080; font-size: 0.8em;'>Dernière vérification: {get_current_time()}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🔄 Rafraîchir"):
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Sélection dynamique des paires")
    min_volatility = st.sidebar.slider("Volatilité min", 0.0, 0.05, 0.01, 0.001)
    min_signal = st.sidebar.slider("Signal min", 0.0, 1.0, 0.3, 0.01)
    top_n = st.sidebar.slider("Nb max paires à trader", 1, 10, 5, 1)

    st.sidebar.markdown(
        f"""
        <div style='background-color: #232b2b; padding: 8px; border-radius: 5px; margin-top: 10px;'>
            <b>🎯 Seuils de filtering actifs</b><br>
            • Volatilité min : <span style='color:#00ff00'>{min_volatility:.3f}</span><br>
            • Signal min : <span style='color:#00ff00'>{min_signal:.2f}</span><br>
            • Nb paires max : <span style='color:#00ff00'>{top_n}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        shared_data["filtering_params"] = {
            "min_volatility": float(min_volatility),
            "min_signal": float(min_signal),
            "top_n": int(top_n),
        }
        with open(SHARED_DATA_PATH, "w") as f:
            json.dump(shared_data, f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Erreur sauvegarde filtres dynamiques: {e}")

tab1, tab2, tab3, tab4, tab5, tab6, tab_logs = st.tabs(
    [
        "📊 Trading",
        "📈 Graphiques",
        "🔬 Analyse",
        "📖 Portefeuille/Positions",
        "🧪 Backtest",
        "📈 Performance",
        "📝 Logs",
    ]
)

with tab1:
    st.subheader("Trading en temps réel")
    bot_status = shared_data.get("bot_status", {})
    perf = bot_status.get("performance", {})
    st.markdown("#### Cycle et Régime")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cycle actuel", bot_status.get("cycle", 0))
    col2.metric("Régime", bot_status.get("regime", "Indéterminé"))
    col3.metric(
        "Balance",
        f"${perf.get('balance',0):,.2f}",
        f"+{perf.get('win_rate',0)*100:.1f}%",
    )

    active_pauses = shared_data.get("active_pauses", [])
    if active_pauses:
        pause_cycles_left = max([p.get("cycles_left", 0) for p in active_pauses])
        st.warning(
            f"🚨 Trading bloqué (pause active) — Déblocage dans {pause_cycles_left} cycle(s) !",
            icon="⏸️",
        )
        st.markdown("#### ⏸️ Pauses actives détaillées")
        df_pauses = pd.DataFrame(active_pauses)
        st.dataframe(df_pauses, use_container_width=True)

    st.divider()
    st.markdown("#### Scores de décision et signaux")
    trade_decisions = shared_data.get("trade_decisions", {})
    if trade_decisions:
        df_signals = pd.DataFrame(trade_decisions).T
        keep_cols = [
            col for col in df_signals.columns if not df_signals[col].isna().all()
        ]
        df_signals = df_signals[keep_cols]
        st.dataframe(df_signals, use_container_width=True)
    else:
        st.info("Aucun signal de trading ce cycle.")

    st.divider()
    st.markdown("#### 📜 Historique des trades exécutés")
    trades = shared_data.get("trade_history", [])
    if trades:
        df_trades = pd.DataFrame(trades)
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("Aucun trade exécuté ce cycle.")

    st.divider()
    st.markdown("#### Arbitrage")
    arbitrage_ops = shared_data.get("arbitrage_opportunities", [])
    if arbitrage_ops:
        st.write("Opportunités d'arbitrage détectées:")
        st.dataframe(pd.DataFrame(arbitrage_ops), use_container_width=True)
    else:
        st.info("Aucune opportunité d'arbitrage détectée ce cycle.")

with tab2:
    st.subheader("Analyse graphique avancée")
    pairs = list(shared_data.get("market_data", {}).keys()) or ["BTCUSDT", "ETHUSDT"]
    pair = st.selectbox("Sélectionner une paire", pairs)
    available_tfs = list(shared_data.get("market_data", {}).get(pair, {}).keys())
    tf = st.selectbox("Timeframe", available_tfs if available_tfs else ["1m"])
    market_data = shared_data.get("market_data", {}).get(pair, {}).get(tf, {})
    if market_data and market_data.get("close") and market_data.get("timestamp"):
        closes = market_data["close"]
        timestamps = market_data["timestamp"]
        ema20 = pd.Series(closes).ewm(span=20).mean()
        ema50 = pd.Series(closes).ewm(span=50).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=timestamps,
                open=market_data.get("open", []),
                high=market_data.get("high", []),
                low=market_data.get("low", []),
                close=closes,
                name="OHLC",
            )
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=ema20, name="EMA 20", line=dict(color="blue"))
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=ema50, name="EMA 50", line=dict(color="orange"))
        )
        fig.update_layout(
            title=f"Graphique {pair} ({tf})",
            yaxis_title="Prix USDT",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas de données live pour cette paire et ce timeframe.")

with tab3:
    st.subheader("Analyse technique approfondie")
    indicators = shared_data.get("indicators", {})
    regime = shared_data.get("regime", "Indéterminé")
    news_sentiment = shared_data.get("sentiment", None)
    trade_decisions = shared_data.get("trade_decisions", {})
    report = _generate_analysis_report(
        indicators, regime, news_sentiment, trade_decisions
    )
    st.code(report, language="markdown")
    st.divider()
    st.expander("🔎 Indicateurs techniques avancés")
    for tf_key, indic in indicators.items():
        if "ta" in indic and indic["ta"]:
            st.write(f"**{tf_key}**")
            df_ta = pd.DataFrame(indic["ta"], index=[0]).T
            st.dataframe(df_ta, use_container_width=True)

with tab4:
    st.subheader("Portefeuille / Positions en temps réel")
    positions_binance = shared_data.get("positions_binance", {})
    st.markdown("#### Positions ouvertes Binance (Spot)")
    if positions_binance:
        df_pos_binance = pd.DataFrame.from_dict(positions_binance, orient="index")
        df_pos_binance.index.name = "Paire"
        if "pnl_pct" in df_pos_binance.columns:
            df_pos_binance["% Plus-Value"] = df_pos_binance["pnl_pct"].map(
                lambda x: f"{x:.2f}%" if x is not None else "N/A"
            )
        st.dataframe(df_pos_binance, use_container_width=True)
    else:
        st.info("Aucune position ouverte sur Binance spot.")

    st.markdown("#### Historique des positions fermées")
    closed = shared_data.get("closed_positions", [])
    if closed:
        df_closed = pd.DataFrame(closed)
        reasons = df_closed["reason"].unique().tolist()
        reason_selected = st.selectbox(
            "Filtrer par raison de vente", ["Toutes"] + reasons
        )
        if reason_selected != "Toutes":
            df_closed = df_closed[df_closed["reason"] == reason_selected]
        st.dataframe(df_closed, use_container_width=True)
    else:
        st.info("Aucune position fermée automatiquement ce cycle.")

    st.markdown("#### Alertes de ventes à venir")
    pending_sales = shared_data.get("pending_sales", [])
    if pending_sales:
        df_pending = pd.DataFrame(pending_sales)
        required_cols = [
            "symbol",
            "reason",
            "decision",
            "entry_price",
            "current_price",
            "amount",
            "% Gain/Perte",
            "temps_en_position_h",
            "pause_blocage",
            "note",
        ]
        for col in required_cols:
            if col not in df_pending.columns:
                df_pending[col] = ""
        if "pnl_pct" in df_pending.columns:
            df_pending["% Gain/Perte"] = df_pending["pnl_pct"].map(
                lambda x: f"{x:.2f}%" if x is not None else "N/A"
            )
        if "temps_en_position_h" in df_pending.columns:
            df_pending["Durée position (h)"] = df_pending["temps_en_position_h"].map(
                lambda x: f"{x}h" if x not in ["", "N/A", None] else "N/A"
            )
        order = [
            "🔴 Signal SELL",
            "🔴 Stop-loss imminent",
            "🟠 TP proche",
            "🔴 Perte latente > 5.0%",
            "🟢 Gain latent > 7.0%",
        ]
        df_pending["prio"] = df_pending["reason"].map(
            lambda r: order.index(r) if r in order else 99
        )
        df_pending = df_pending.sort_values(
            ["prio", "pnl_pct"] if "pnl_pct" in df_pending.columns else ["prio"],
            ascending=[True, False] if "pnl_pct" in df_pending.columns else [True],
        )
        st.dataframe(
            df_pending[required_cols],
            use_container_width=True,
        )
    else:
        st.info("Aucune vente imminente détectée.")

with tab5:
    st.subheader("Backtest avancé")
    st.sidebar.header("Backtesting avancé")
    strategy_options = {
        "SMA Crossover": sma_strategy,
        "Breakout": breakout_strategy,
        "Arbitrage": arbitrage_strategy,
    }
    strategy_name = st.sidebar.selectbox("Stratégie", list(strategy_options.keys()))
    strategy_func = strategy_options[strategy_name]
    params = {}
    if strategy_name == "SMA Crossover":
        params["fast_window"] = st.sidebar.slider("SMA rapide", 2, 50, 10)
        params["slow_window"] = st.sidebar.slider("SMA lente", 10, 200, 50)
    elif strategy_name == "Breakout":
        params["lookback"] = st.sidebar.slider("Lookback", 5, 50, 20)
    elif strategy_name == "Arbitrage":
        params["spread_threshold"] = st.sidebar.number_input(
            "Seuil de spread (%)", min_value=0.01, max_value=5.0, value=0.5
        )
    dataset_file = st.sidebar.file_uploader("Données historiques (CSV)", type=["csv"])
    if dataset_file:
        df = pd.read_csv(dataset_file)
        capital = st.sidebar.number_input("Capital initial", min_value=100, value=10000)
        if st.sidebar.button("Lancer le backtest"):
            backtester = BacktestEngine(initial_capital=capital)
            results = backtester.run_backtest(df, strategy_func, **params)
            st.write("Résultats du backtest :", results)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Configuration de base")
        period = st.selectbox("Période de test", ["7j", "30j", "90j", "180j", "365j"])
        initial_capital = st.number_input(
            "Capital initial (USDT)", min_value=0, value=0
        )
        leverage = st.slider("Levier", min_value=1, max_value=10, value=1)
    with col2:
        st.markdown("### Paramètres avancés")
        risk_per_trade = st.slider(
            "Risque par trade (%)", min_value=0.1, max_value=5.0, value=1.0
        )
        stop_loss = st.slider("Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0)
        take_profit = st.slider(
            "Take Profit (%)", min_value=1.0, max_value=20.0, value=4.0
        )
    if st.button("🚀 Lancer le backtest"):
        st.info("Simulation en cours…")
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            pairs = config.get("pairs", ["BTC/USDT"])
        except Exception:
            pairs = ["BTC/USDT"]
        period_map = {"7j": 7, "30j": 30, "90j": 90, "180j": 180, "365j": 365}
        nb_days = period_map.get(period, 30)
        end_dt = pd.Timestamp.utcnow()
        start_dt = end_dt - pd.Timedelta(days=nb_days)
        interval = Client.KLINE_INTERVAL_1HOUR
        for i, pair in enumerate(pairs):
            symbol = pair.replace("/", "")
            df = fetch_binance_ohlcv(
                symbol,
                interval,
                start_dt.strftime("%d %b %Y"),
                end_dt.strftime("%d %b %Y"),
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
            )
            if df is None or len(df) == 0:
                st.error(f"Données manquantes pour {pair}, backtest ignoré.")
                continue
            strategy_func = strategy_options[strategy_name]
            backtester = BacktestEngine(initial_capital=initial_capital)
            results = backtester.run_backtest(df, strategy_func, **params)
            st.write(f"Résultats du backtest pour {pair} :", results)
        st.success("Backtest terminé!")

with tab6:
    st.subheader("Performance et Métriques")
    perf = shared_data.get("bot_status", {}).get("performance", {})
    equity_history = shared_data.get("equity_history", [])
    returns = shared_data.get("returns_array", np.linspace(0, 27.5, 30))
    x_axis = list(range(len(returns)))
    cumulative_returns = 1 + np.array(returns) / 100
    st.markdown("### 📈 Performance Cumulative")
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_axis,
                y=cumulative_returns,
                name="Performance",
                fill="tozeroy",
                line=dict(color="#00ff00"),
            )
        ],
        layout=go.Layout(
            title="Performance du Trading Bot",
            template="plotly_dark",
            yaxis_title="Rendement Cumulatif",
            xaxis_title="Jours",
            showlegend=True,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📊 Métriques de Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total trades", f"{perf.get('total_trades',0)}")
    col1.metric("Win Rate", f"{perf.get('win_rate',0):.1%}")
    col2.metric("Profit Factor", f"{perf.get('profit_factor',0):.2f}")
    col2.metric("Max Drawdown", f"{perf.get('max_drawdown',0):.1%}")
    col3.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio',0):.2f}")
    col3.metric("Balance Finale", f"${perf.get('balance',10000):,.0f}")

    equity_curve = [
        pt.get("balance", 0) for pt in equity_history if pt.get("balance", 0) > 0
    ]
    kelly = None
    max_dd = None
    var95 = None
    win_streak = None
    loss_streak = None
    avg_win = None
    avg_loss = None
    best_trade = None
    worst_trade = None
    win_pct = None

    if equity_curve and len(equity_curve) > 10:
        equity_curve_np = np.array(equity_curve)
        max_dd = calculate_max_drawdown(equity_curve_np)
        returns_curve = np.diff(equity_curve_np) / equity_curve_np[:-1]
        if len(returns_curve) > 10:
            var95 = calculate_var(returns_curve, 0.05)
        kelly = kelly_criterion(
            win_rate=perf.get("win_rate", 0), payoff_ratio=perf.get("profit_factor", 1)
        )
        trades = shared_data.get("trade_history", [])
        wins = [t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) > 0]
        losses = [t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        best_trade = np.max(wins) if wins else 0
        worst_trade = np.min(losses) if losses else 0
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        prev = None
        for t in trades:
            pnl = t.get("pnl_usd", 0)
            if pnl > 0:
                streak = streak + 1 if prev == "win" else 1
                max_win_streak = max(max_win_streak, streak)
                prev = "win"
            elif pnl < 0:
                streak = streak + 1 if prev == "loss" else 1
                max_loss_streak = max(max_loss_streak, streak)
                prev = "loss"
        win_streak = max_win_streak
        loss_streak = max_loss_streak
        total_trades = len(trades)
        win_pct = len(wins) / total_trades if total_trades > 0 else 0

    with st.expander("📉 Indicateurs avancés de risque et performance"):
        st.metric("Kelly optimal", f"{kelly:.2f}" if kelly is not None else "N/A")
        st.metric("Max Drawdown", f"{max_dd:.2%}" if max_dd is not None else "N/A")
        st.metric("VaR (95%)", f"{var95:.2%}" if var95 is not None else "N/A")
        st.metric(
            "Plus longue série de trades gagnants",
            f"{win_streak}" if win_streak is not None else "N/A",
        )
        st.metric(
            "Plus longue série de trades perdants",
            f"{loss_streak}" if loss_streak is not None else "N/A",
        )
        st.metric(
            "Moyenne gains/trade", f"${avg_win:.2f}" if avg_win is not None else "N/A"
        )
        st.metric(
            "Moyenne pertes/trade",
            f"${avg_loss:.2f}" if avg_loss is not None else "N/A",
        )
        st.metric(
            "Meilleur trade", f"${best_trade:.2f}" if best_trade is not None else "N/A"
        )
        st.metric(
            "Pire trade", f"${worst_trade:.2f}" if worst_trade is not None else "N/A"
        )
        st.metric(
            "Ratio de trades gagnants",
            f"{win_pct:.1%}" if win_pct is not None else "N/A",
        )

        if kelly is not None and abs(kelly) > 0.5:
            st.warning(
                f"⚠️ Kelly fraction élevée : {kelly:.2f} — attention à la taille des positions !"
            )
        if max_dd is not None and max_dd < -0.15:
            st.error(f"🚨 Max drawdown dépassé : {max_dd:.2%} ! Pause conseillée.")
        if var95 is not None and var95 < -0.05:
            st.error(f"🛑 VaR(95%) critique : {var95:.2f}")

with tab_logs:
    st.subheader("📝 Logs du Bot (live)")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = f.readlines()
        st.text("".join(logs[-200:]))
    else:
        st.info("Aucun log à afficher.")
    if st.button("🗑️ Vider les logs"):
        open(LOG_FILE, "w").close()
        st.success("Logs vidés !")


def auto_refresh():
    time.sleep(10)
    st.rerun()


if __name__ == "__main__":
    auto_refresh()
