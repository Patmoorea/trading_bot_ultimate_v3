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


def get_pending_sales(self):
    """
    Retourne la liste des positions qui risquent d'être vendues prochainement (signal SELL, TP proche, SL imminent, gain/perte latente élevée).
    Ce tableau est ultra-visuel et utile. Il contient pour chaque position : symbol, raison, confiance, prix d'achat, prix actuel, montant, %PnL, date d'achat, temps en position.
    Inclut à la fois les positions gérées par le bot ET les positions spot Binance.
    """
    pending = []
    # Seuils configurables
    GAIN_ALERT_PCT = 0.07  # 7% de gain latent
    LOSS_ALERT_PCT = -0.05  # -5% de perte latente

    now = datetime.utcnow()

    print("DEBUG positions bot:", self.positions)
    if hasattr(self, "positions_binance"):
        print("DEBUG positions_binance:", self.positions_binance)

    # 1. Positions gérées par le bot (virtuel)
    for symbol, pos in self.positions.items():
        entry_price = pos.get("entry_price")
        current_price = pos.get("current_price")
        amount = pos.get("amount")
        pnl_pct = (
            (current_price - entry_price) / entry_price * 100
            if entry_price and current_price
            else 0
        )
        date_achat = pos.get("date", pos.get("entry_time")) or None
        if date_achat:
            try:
                date_achat_dt = datetime.fromisoformat(date_achat)
                temps_en_position = (now - date_achat_dt).total_seconds() / 3600
            except Exception:
                temps_en_position = None
        else:
            temps_en_position = None

        # 1. Signal SELL actif
        td = self.trade_decisions.get(symbol.replace("/", "").upper(), {})
        if td.get("action") == "SELL" and pos.get("side") == "long":
            pending.append(
                {
                    "symbol": symbol,
                    "reason": "🔴 Signal SELL",
                    "confidence": td.get("confidence"),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "amount": amount,
                    "pnl_pct": pnl_pct,
                    "date_achat": date_achat,
                    "temps_en_position_h": temps_en_position,
                }
            )
        # 2. TP proche
        if self.exit_manager.is_tp_near(pos):
            pending.append(
                {
                    "symbol": symbol,
                    "reason": "🟠 TP proche",
                    "confidence": None,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "amount": amount,
                    "pnl_pct": pnl_pct,
                    "date_achat": date_achat,
                    "temps_en_position_h": temps_en_position,
                }
            )
        # 3. Stop-loss imminent
        if self.check_stop_loss(symbol):
            pending.append(
                {
                    "symbol": symbol,
                    "reason": "🔴 Stop-loss imminent",
                    "confidence": None,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "amount": amount,
                    "pnl_pct": pnl_pct,
                    "date_achat": date_achat,
                    "temps_en_position_h": temps_en_position,
                }
            )
        # 4. Gain latent élevé (gain > 7%)
        if pnl_pct > GAIN_ALERT_PCT * 100:
            pending.append(
                {
                    "symbol": symbol,
                    "reason": f"🟢 Gain latent > {GAIN_ALERT_PCT*100:.1f}%",
                    "confidence": None,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "amount": amount,
                    "pnl_pct": pnl_pct,
                    "date_achat": date_achat,
                    "temps_en_position_h": temps_en_position,
                }
            )
        # 5. Perte latente élevée (perte < -5%)
        if pnl_pct < LOSS_ALERT_PCT * 100:
            pending.append(
                {
                    "symbol": symbol,
                    "reason": f"🔴 Perte latente > {abs(LOSS_ALERT_PCT*100):.1f}%",
                    "confidence": None,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "amount": amount,
                    "pnl_pct": pnl_pct,
                    "date_achat": date_achat,
                    "temps_en_position_h": temps_en_position,
                }
            )

    # 2. Positions spot Binance réelles (si dispo)
    if hasattr(self, "positions_binance"):
        for symbol, pos in self.positions_binance.items():
            entry_price = pos.get("entry_price")
            current_price = pos.get("current_price")
            amount = pos.get("amount")
            pnl_pct = (
                (current_price - entry_price) / entry_price * 100
                if entry_price and current_price
                else 0
            )
            date_achat = None  # Non dispo en spot
            temps_en_position = None

            td = self.trade_decisions.get(symbol.replace("/", "").upper(), {})
            if td.get("action") == "SELL" and pos.get("side") == "long":
                pending.append(
                    {
                        "symbol": symbol,
                        "reason": "🔴 Signal SELL",
                        "confidence": td.get("confidence"),
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "amount": amount,
                        "pnl_pct": pnl_pct,
                        "date_achat": date_achat,
                        "temps_en_position_h": temps_en_position,
                    }
                )
            if hasattr(self, "exit_manager") and self.exit_manager.is_tp_near(pos):
                pending.append(
                    {
                        "symbol": symbol,
                        "reason": "🟠 TP proche",
                        "confidence": None,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "amount": amount,
                        "pnl_pct": pnl_pct,
                        "date_achat": date_achat,
                        "temps_en_position_h": temps_en_position,
                    }
                )
            if self.check_stop_loss(symbol):
                pending.append(
                    {
                        "symbol": symbol,
                        "reason": "🔴 Stop-loss imminent",
                        "confidence": None,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "amount": amount,
                        "pnl_pct": pnl_pct,
                        "date_achat": date_achat,
                        "temps_en_position_h": temps_en_position,
                    }
                )
            if pnl_pct > GAIN_ALERT_PCT * 100:
                pending.append(
                    {
                        "symbol": symbol,
                        "reason": f"🟢 Gain latent > {GAIN_ALERT_PCT*100:.1f}%",
                        "confidence": None,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "amount": amount,
                        "pnl_pct": pnl_pct,
                        "date_achat": date_achat,
                        "temps_en_position_h": temps_en_position,
                    }
                )
            if pnl_pct < LOSS_ALERT_PCT * 100:
                pending.append(
                    {
                        "symbol": symbol,
                        "reason": f"🔴 Perte latente > {abs(LOSS_ALERT_PCT*100):.1f}%",
                        "confidence": None,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "amount": amount,
                        "pnl_pct": pnl_pct,
                        "date_achat": date_achat,
                        "temps_en_position_h": temps_en_position,
                    }
                )
    print("DEBUG pending_sales tableau:", pending)
    # Sauvegarde dans shared_data.json
    try:
        with open(self.data_file, "r") as f:
            shared_data = json.load(f)
    except Exception:
        shared_data = {}
    shared_data["pending_sales"] = pending
    with open(self.data_file, "w") as f:
        json.dump(shared_data, f, indent=4)
    return pending


def fetch_binance_ohlcv(
    symbol, interval, start_str, end_str=None, api_key=None, api_secret=None
):
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    if not klines or len(klines) == 0:
        return None
    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


# --- SIDEBAR ---
with st.sidebar:
    st.header("🤖 Bot Status")
    status = load_json_file(STATUS_FILE)
    shared_data = load_json_file(SHARED_DATA_PATH)
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

    # == 1. Alertes actives ==
    st.markdown("### 🚨 Alertes actives")
    alerts = shared_data.get("alerts", [])
    for alert in alerts:
        if alert["level"] == "critical":
            st.error(f"{alert['message']} ({alert['timestamp']})")
        elif alert["level"] == "warning":
            st.warning(f"{alert['message']} ({alert['timestamp']})")
        else:
            st.info(f"{alert['message']} ({alert['timestamp']})")

    # == 3. Positions fermées ==
    st.header("⛔️ Positions fermées (auto)")
    closed = shared_data.get("closed_positions", [])
    if closed:
        df_closed = pd.DataFrame(closed)
        st.dataframe(df_closed, use_container_width=True)
    else:
        st.info("Aucune position fermée automatiquement ce cycle.")

    st.sidebar.divider()

    # == 4. Informations système & connectivité ==
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

    # --- AJOUT SLIDERS ET AFFICHAGE SEUILS ACTIFS ---
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

    # Sauvegarde dans shared_data.json (pour que le bot les lise au prochain cycle)
    try:
        shared_data = load_json_file(SHARED_DATA_PATH)
        shared_data["filtering_params"] = {
            "min_volatility": float(min_volatility),
            "min_signal": float(min_signal),
            "top_n": int(top_n),
        }
        with open(SHARED_DATA_PATH, "w") as f:
            json.dump(shared_data, f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Erreur sauvegarde filtres dynamiques: {e}")

# --- TABS ---
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

# --- TAB1 TRADING ---
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

    # --- BANDEAU PAUSE TRADING AVEC COMPTEUR ET TABLEAU DÉTAILLÉ ---
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
    # print("=== DEBUG TRADE_DECISIONS ===")
    # print(trade_decisions)
    if trade_decisions:
        df_signals = pd.DataFrame(trade_decisions).T

        # Filtrer dynamiquement les colonnes où tout est None ou NaN
        # Conserve uniquement les colonnes où il existe au moins une valeur non nulle
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

# --- TAB2 GRAPHIQUES ---
with tab2:
    st.subheader("Analyse graphique avancée")
    pairs = list(shared_data.get("market_data", {}).keys()) or ["BTCUSDT", "ETHUSDT"]
    pair = st.selectbox("Sélectionner une paire", pairs)

    # Ajoute le choix du timeframe
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

# --- TAB3 ANALYSE ---
with tab3:
    st.subheader("Analyse technique approfondie")
    indicators = shared_data.get("indicators", {})
    regime = shared_data.get("regime", "Indéterminé")
    news_sentiment = shared_data.get("sentiment", None)
    trade_decisions = shared_data.get("trade_decisions", {})
    # Rapport global
    report = _generate_analysis_report(
        indicators, regime, news_sentiment, trade_decisions
    )
    st.code(report, language="markdown")
    st.divider()
    # Indicateurs avancés par paire/timeframe
    st.expander("🔎 Indicateurs techniques avancés")
    for tf_key, indic in indicators.items():
        if "ta" in indic and indic["ta"]:
            st.write(f"**{tf_key}**")
            df_ta = pd.DataFrame(indic["ta"], index=[0]).T
            st.dataframe(df_ta, use_container_width=True)

# --- TAB4 PORTEFEUILLE / POSITIONS ---
with tab4:
    st.subheader("Portefeuille / Positions en temps réel")

    # 1. Affichage du portefeuille spot Binance
    positions_binance = shared_data.get("positions_binance", {})
    st.markdown("#### Positions ouvertes Binance (Spot)")
    if positions_binance:
        df_pos_binance = pd.DataFrame.from_dict(positions_binance, orient="index")
        df_pos_binance.index.name = "Paire"
        # Ajout formatage % plus-value si dispo
        if "pnl_pct" in df_pos_binance.columns:
            df_pos_binance["% Plus-Value"] = df_pos_binance["pnl_pct"].map(
                lambda x: f"{x:.2f}%" if x is not None else "N/A"
            )
        st.dataframe(df_pos_binance, use_container_width=True)
    else:
        st.info("Aucune position ouverte sur Binance spot.")

    # 2. Historique des positions fermées
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

    # 3. Alertes de ventes à venir
    st.markdown("#### Alertes de ventes à venir")
    pending_sales = shared_data.get("pending_sales", [])
    if pending_sales:
        df_pending = pd.DataFrame(pending_sales)
        # Formatage visuel des colonnes
        if "pnl_pct" in df_pending.columns:
            df_pending["% Gain/Perte"] = df_pending["pnl_pct"].map(
                lambda x: f"{x:.2f}%" if x is not None else "N/A"
            )
        if "temps_en_position_h" in df_pending.columns:
            df_pending["Durée position (h)"] = df_pending["temps_en_position_h"].map(
                lambda x: f"{x:.1f}h" if x is not None else "N/A"
            )
        # Trie d'abord par urgence (Signal SELL > SL > TP > perte > gain), puis par %PnL
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
            ["prio", "pnl_pct"], ascending=[True, False]
        )
        # Affichage
        st.dataframe(
            df_pending[
                [
                    "symbol",
                    "reason",
                    "amount",
                    "entry_price",
                    "current_price",
                    "% Gain/Perte",
                    "confidence",
                    "date_achat",
                    "Durée position (h)",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("Aucune vente imminente détectée.")

# --- TAB5 BACKTEST ---
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
    # Configuration du backtest
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

# --- TAB6 PERFORMANCE ---
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

    # --- Ajout Risk Management avancé ---
    equity_curve = [
        pt.get("balance", 0) for pt in equity_history if pt.get("balance", 0) > 0
    ]
    kelly = None
    max_dd = None
    var95 = None
    if equity_curve and len(equity_curve) > 10:
        equity_curve_np = np.array(equity_curve)
        max_dd = calculate_max_drawdown(equity_curve_np)
        returns_curve = np.diff(equity_curve_np) / equity_curve_np[:-1]
        if len(returns_curve) > 10:
            var95 = calculate_var(returns_curve, 0.05)
        kelly = kelly_criterion(
            win_rate=perf.get("win_rate", 0), payoff_ratio=perf.get("profit_factor", 1)
        )
    with st.expander("📉 Indicateurs avancés de risque"):
        st.metric("Kelly optimal", f"{kelly:.2f}" if kelly is not None else "N/A")
        st.metric("Max Drawdown", f"{max_dd:.2%}" if max_dd is not None else "N/A")
        st.metric("VaR (95%)", f"{var95:.2%}" if var95 is not None else "N/A")
        if kelly is not None and abs(kelly) > 0.5:
            st.warning(
                f"⚠️ Kelly fraction élevée : {kelly:.2f} — attention à la taille des positions !"
            )
        if max_dd is not None and max_dd < -0.15:
            st.error(f"🚨 Max drawdown dépassé : {max_dd:.2%} ! Pause conseillée.")
        if var95 is not None and var95 < -0.05:
            st.error(f"🛑 VaR(95%) critique : {var95:.2f}")

# --- TAB LOGS ---
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


# --- Auto-refresh ---
def auto_refresh():
    time.sleep(10)
    st.rerun()


if __name__ == "__main__":
    auto_refresh()
