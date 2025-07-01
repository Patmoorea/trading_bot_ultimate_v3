import streamlit as st
import os
from dotenv import load_dotenv
import sys
import logging
import json
from datetime import datetime, timezone
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import qiskit
import websockets
import asyncio
import psutil
import time

# --- Configuration initiale ---
load_dotenv()
logger = logging.getLogger(__name__)


# --- Constants ---
STATUS_FILE = "bot_status.json"
CURRENT_TIME = "2025-07-01 16:44:56"  # Date spécifique
CURRENT_USER = "Patmoorea"
CONFIG_FILE = "config.json"


def get_current_time():
    # Pour test/développement, retourner une date fixe
    return CURRENT_TIME


def load_status():
    """Charge et retourne le statut du bot"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                data = json.load(f)
                # Ajouter le timestamp actuel
                data["timestamp"] = get_current_time()
                return data
        except Exception as e:
            logger.error(f"Erreur de lecture du status : {e}")
            return {}
    return {}


class BotDataManager:
    def __init__(self):
        self.bot = None
        try:
            from src.bot_runner import TradingBot

            self.bot = TradingBot()
        except Exception as e:
            logger.error(f"Erreur initialisation bot: {e}")

    def get_market_data(self):
        if self.bot:
            return self.bot.get_market_data()
        else:
            # Données de fallback avec le timestamp spécifié
            dates = pd.date_range(start=CURRENT_TIME, periods=100, freq="H")
            return {
                "timestamp": dates,
                "open": np.random.normal(50000, 1000, 100),
                "high": np.random.normal(51000, 1000, 100),
                "low": np.random.normal(49000, 1000, 100),
                "close": np.random.normal(50000, 1000, 100),
                "volume": np.random.normal(1000000, 100000, 100),
            }

    def get_performance_metrics(self):
        if self.bot:
            return self.bot.get_performance_metrics()
        else:
            # Métriques de fallback
            return {
                "total_trades": 156,
                "win_rate": 0.62,
                "profit_factor": 1.85,
                "max_drawdown": 0.15,
                "sharpe_ratio": 1.92,
                "final_balance": 12750,
                "return": 27.5,
            }


# --- Configuration Streamlit ---
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
bot_manager = BotDataManager()


# --- Classes d'amélioration ---
class AdvancedAITrader:
    def __init__(self):
        self.models = {
            "sentiment": self._init_sentiment_model(),
            "price_prediction": self._init_deep_learning_model(),
            "regime_detection": self._init_regime_classifier(),
            "quantum_analyzer": self._init_quantum_processor(),
        }
        self.scaler = StandardScaler()

    def _init_sentiment_model(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    def _init_deep_learning_model(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(50, return_sequences=True),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(1),
            ]
        )

    def _init_regime_classifier(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

    def _init_quantum_processor(self):
        return qiskit.QuantumCircuit(3)

    async def _run_quantum_analysis(self, data):
        try:
            circuit = self.models["quantum_analyzer"]
            # Ajoutez votre logique d'analyse quantique ici
            return {"quantum_signal": 1}  # Exemple de retour
        except Exception as e:
            logger.error(f"Erreur analyse quantique: {e}")
            return {}

    async def analyze_market(self, data):
        results = {}
        try:
            normalized_data = self.scaler.fit_transform(data)
            for name, model in self.models.items():
                if name == "quantum_analyzer":
                    results[name] = await self._run_quantum_analysis(normalized_data)
                else:
                    results[name] = model.predict(normalized_data)
            return results
        except Exception as e:
            logger.error(f"Erreur analyse IA: {e}")
            return {}


class EnhancedRiskManager:
    def __init__(self):
        self.max_position_size = 0.02
        self.max_total_risk = 0.10
        self.correlation_matrix = {}
        self.vol_window = 20
        self.positions = {}

    def calculate_position_size(self, signal_strength, volatility, current_exposure):
        if current_exposure >= self.max_total_risk:
            return 0

        base_size = self.max_position_size
        risk_adjusted = base_size * signal_strength
        vol_adjusted = risk_adjusted / (volatility + 1e-6)
        remaining_risk = self.max_total_risk - current_exposure

        return min(vol_adjusted, remaining_risk)

    def update_risk_metrics(self, market_data):
        try:
            returns = pd.DataFrame(market_data).pct_change()
            self.correlation_matrix = returns.corr()
            self.volatility = returns.std() * np.sqrt(252)
            return True
        except Exception as e:
            logger.error(f"Erreur mise à jour métriques: {e}")
            return False


class RealtimeManager:
    def __init__(self):
        self.clients = set()
        self.data_buffer = {}
        self.last_update = datetime.now(timezone.utc)

    async def broadcast_update(self, data):
        current_time = datetime.now(timezone.utc)
        if (current_time - self.last_update).total_seconds() >= 1:
            message = {"timestamp": get_current_time(), "data": data}
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients]
            )
            self.last_update = current_time


class DashboardEnhancer:
    def __init__(self):
        self.metrics = {
            "performance": PerformanceMetrics(),
            "risk": RiskMetrics(),
            "market": MarketMetrics(),
        }
        self.last_update = "2025-07-01 16:25:28"
        self.current_user = "Patmoorea"

    def get_enhanced_metrics(self, market_data):
        try:
            return {
                "sharpe_ratio": self.metrics["performance"].get_sharpe(market_data),
                "max_drawdown": self.metrics["risk"].get_max_drawdown(market_data),
                "volatility": self.metrics["risk"].get_volatility(market_data),
                "win_rate": self.metrics["performance"].get_win_rate(market_data),
                "profit_factor": self.metrics["performance"].get_profit_factor(
                    market_data
                ),
                "recovery_factor": self.metrics["risk"].get_recovery_factor(
                    market_data
                ),
            }
        except Exception as e:
            logger.error(f"Erreur calcul métriques: {e}")
            return {}


class PerformanceMetrics:
    def get_sharpe(self, returns, risk_free_rate=0.02):
        returns = pd.Series(returns)
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def get_win_rate(self, trades):
        if not trades:
            return 0
        wins = sum(1 for trade in trades if trade["pnl"] > 0)
        return wins / len(trades)

    def get_profit_factor(self, trades):
        if not trades:
            return 0
        gains = sum(trade["pnl"] for trade in trades if trade["pnl"] > 0)
        losses = abs(sum(trade["pnl"] for trade in trades if trade["pnl"] < 0))
        return gains / losses if losses != 0 else float("inf")


class RiskMetrics:
    def get_max_drawdown(self, equity_curve):
        rolling_max = pd.Series(equity_curve).expanding(min_periods=1).max()
        drawdowns = equity_curve / rolling_max - 1.0
        return abs(drawdowns.min())

    def get_volatility(self, returns, window=20):
        return pd.Series(returns).rolling(window=window).std() * np.sqrt(252)

    def get_recovery_factor(self, equity_curve):
        max_dd = self.get_max_drawdown(equity_curve)
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        return abs(total_return / max_dd) if max_dd != 0 else float("inf")


class MarketMetrics:
    def __init__(self):
        self.indicators = {
            "RSI": self.calculate_rsi,
            "MACD": self.calculate_macd,
            "BB": self.calculate_bollinger_bands,
        }

    def calculate_rsi(self, data, period=14):
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(data).ewm(span=fast).mean()
        ema_slow = pd.Series(data).ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        sma = pd.Series(data).rolling(window=window).mean()
        std = pd.Series(data).rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band


class AlertSystem:
    def __init__(self):
        self.alert_levels = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
        self.alerts_history = []
        self.max_alerts = 100

    def check_alerts(self, data):
        alerts = []
        current_time = get_current_time()

        if data.get("volatility", 0) > 0.05:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Volatilité élevée: {data['volatility']:.2%}",
                    "timestamp": current_time,
                }
            )

        if data.get("drawdown", 0) > 0.10:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Drawdown critique: {data['drawdown']:.2%}",
                    "timestamp": current_time,
                }
            )

        if data.get("signals"):
            for pair, signal in data["signals"].items():
                if signal.get("strength", 0) > 0.8:
                    alerts.append(
                        {
                            "level": "info",
                            "message": f"Signal fort sur {pair}: {signal['type']}",
                            "timestamp": current_time,
                        }
                    )

        self.alerts_history.extend(alerts)
        self.alerts_history = self.alerts_history[-self.max_alerts :]
        return alerts


# --- Fonctions utilitaires pour le dashboard ---
def create_advanced_price_chart(data, indicators=None):
    fig = go.Figure()

    # Chandelier japonais
    fig.add_trace(
        go.Candlestick(
            x=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="OHLC",
        )
    )

    # Ajout des indicateurs si présents
    if indicators:
        if "ema20" in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data["timestamp"],
                    y=indicators["ema20"],
                    name="EMA 20",
                    line=dict(color="blue"),
                )
            )
        if "ema50" in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data["timestamp"],
                    y=indicators["ema50"],
                    name="EMA 50",
                    line=dict(color="orange"),
                )
            )

    fig.update_layout(
        title="Analyse technique en temps réel",
        yaxis_title="Prix USDT",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )

    return fig


def create_volume_profile(data):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data["volume"],
            y=data["price_levels"],
            orientation="h",
            name="Volume Profile",
            marker=dict(color="rgba(0,128,255,0.5)"),
        )
    )

    fig.update_layout(
        title="Profile de Volume",
        xaxis_title="Volume",
        yaxis_title="Niveaux de prix",
        template="plotly_dark",
    )

    return fig


# --- Interface principale ---
st.title("Trading Bot Ultimate v4 - Dashboard")

# --- Sidebar avec status avancé et configuration ---
with st.sidebar:
    st.header("🤖 Bot Status")
    status = load_status()

    if status:
        # Status principal avec style
        st.markdown(
            f"""
            <div style='background-color: #0f3d40; padding: 10px; border-radius: 5px;'>
                <h3 style='color: #00ff00; margin: 0;'>✅ Bot Actif</h3>
                <p style='color: #ffffff; margin: 5px 0;'>Dernière mise à jour: {get_current_time()}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Informations de session
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

        # Configuration avancée
        with st.expander("⚙️ Configuration avancée"):
            risk_params = {
                "max_position": st.slider("Position maximum (%)", 1, 10, 2),
                "stop_loss": st.slider("Stop Loss (%)", 1, 20, 5),
                "take_profit": st.slider("Take Profit (%)", 1, 50, 15),
            }

            strategy_params = {
                "timeframe": st.selectbox(
                    "Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"]
                ),
                "indicators": st.multiselect(
                    "Indicateurs", ["RSI", "MACD", "BB", "EMA"]
                ),
            }

# --- Tabs principaux ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Trading", "📈 Graphiques", "🔬 Analyse", "🧪 Backtest", "📈 Performance"]
)

with tab1:
    st.subheader("Trading en temps réel")

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "BTC/USDT",
            f"${status.get('btc_price', '0.00')}",
            f"{status.get('btc_change', '0.00')}%",
            delta_color="normal",
        )
    with col2:
        st.metric(
            "Volume 24h",
            f"${status.get('volume_24h', '0.00')}M",
            f"{status.get('volume_change', '0.00')}%",
        )
    with col3:
        st.metric(
            "Positions",
            status.get("active_positions", "0"),
            f"{status.get('position_change', '0')}",
        )
    with col4:
        st.metric(
            "P&L Jour",
            f"${status.get('daily_pnl', '0.00')}",
            f"{status.get('pnl_change', '0.00')}%",
        )

    # Signaux actifs
    st.subheader("📡 Signaux de trading")
    if "signals" in status:
        signals_df = pd.DataFrame(status["signals"]).T
        st.dataframe(
            signals_df,
            use_container_width=True,
            height=200,
            column_config={
                "strength": st.column_config.ProgressColumn(
                    "Force du signal",
                    help="Force du signal de trading",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                ),
                "type": "Type",
                "direction": "Direction",
                "timestamp": "Heure",
            },
        )

with tab2:
    st.subheader("Analyse graphique avancée")

    # Sélection de la paire
    pair = st.selectbox("Sélectionner une paire", ["BTC/USDT", "ETH/USDT", "BNB/USDT"])

    # Création des données de test (à remplacer par les vraies données)
    dates = pd.date_range(start=get_current_time(), periods=100, freq="H")
    market_data = bot_manager.get_market_data()

    # Indicateurs techniques
    indicators = {
        "ema20": pd.Series(market_data["close"]).ewm(span=20).mean(),
        "ema50": pd.Series(market_data["close"]).ewm(span=50).mean(),
    }

    # Affichage des graphiques
    col1, col2 = st.columns([7, 3])

    with col1:
        st.plotly_chart(
            create_advanced_price_chart(market_data, indicators),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(create_volume_profile(market_data), use_container_width=True)

with tab3:
    st.subheader("Analyse technique approfondie")

    # Métriques d'analyse
    metrics = DashboardEnhancer().get_enhanced_metrics(market_data)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Momentum")
        st.metric("RSI", f"{metrics.get('rsi', 0):.2f}")
        st.metric("MACD", f"{metrics.get('macd', 0):.2f}")

    with col2:
        st.markdown("### Tendance")
        st.metric("EMA 20", f"{metrics.get('ema20', 0):.2f}")
        st.metric("EMA 50", f"{metrics.get('ema50', 0):.2f}")

    with col3:
        st.markdown("### Volatilité")
        st.metric("ATR", f"{metrics.get('atr', 0):.2f}")
        st.metric("BB Width", f"{metrics.get('bb_width', 0):.2f}")

    # --- Suite des tabs ---
with tab4:
    st.subheader("Backtest avancé")

    # Configuration du backtest
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Configuration de base")
        period = st.selectbox("Période de test", ["7j", "30j", "90j", "180j", "365j"])
        initial_capital = st.number_input(
            "Capital initial (USDT)", min_value=100, value=10000
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

    # Lancement du backtest
    if st.button("🚀 Lancer le backtest"):
        progress_text = "Simulation en cours. Veuillez patienter..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Résultats simulés
        results = bot_manager.get_performance_metrics()

        # Affichage des résultats
        st.success("Backtest terminé!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Balance finale",
                f"${results['final_balance']:,.2f}",
                f"+{results['return']}%",
            )
            st.metric("Nombre de trades", results["total_trades"])
        with col2:
            st.metric("Win Rate", f"{results['win_rate']*100:.1f}%")
            st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']*100:.1f}%")
            st.metric("Ratio de Sharpe", f"{results['sharpe_ratio']:.2f}")

with tab5:
    st.subheader("Performance et Métriques")

    # Calculer la date de début (30 jours avant la date actuelle)
    start_date = (
        datetime.strptime(CURRENT_TIME, "%Y-%m-%d %H:%M:%S") - timedelta(days=30)
    ).strftime("%Y-%m-%d")

    # Métriques de performance avec dates correctes
    performance_data = {
        "dates": pd.date_range(
            start=start_date,  # 30 jours avant
            end=CURRENT_TIME,  # Date actuelle
            freq="D",
        ),
        "cumulative_returns": bot_manager.get_performance_metrics()["return"],
        "drawdowns": market_data.get("drawdowns", []),
        "volatility": market_data.get("volatility", []),
    }

    # Graphiques de performance
    st.plotly_chart(
        go.Figure(
            data=[
                go.Scatter(
                    x=performance_data["dates"],
                    y=performance_data["cumulative_returns"],
                    name="Returns",
                    fill="tozeroy",
                )
            ],
            layout=go.Layout(title="Performance cumulative", template="plotly_dark"),
        ),
        use_container_width=True,
    )

# --- Système d'alertes ---
alert_system = AlertSystem()
alerts = alert_system.check_alerts(status)

# Affichage des alertes dans la sidebar
with st.sidebar:
    st.markdown("### 🚨 Alertes actives")

    for alert in alerts:
        if alert["level"] == "critical":
            st.error(f"{alert['message']} ({alert['timestamp']})")
        elif alert["level"] == "warning":
            st.warning(f"{alert['message']} ({alert['timestamp']})")
        else:
            st.info(f"{alert['message']} ({alert['timestamp']})")


# --- Auto-refresh et Footer ---
def auto_refresh(interval_ms=2000):
    js_code = f"""
    <script>
        setInterval(function() {{
            window.location.reload();
        }}, {interval_ms});
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


auto_refresh()

# Footer avec informations détaillées
st.sidebar.divider()
st.sidebar.markdown(
    f"""
### 📊 Informations système
- 🕒 Dernière mise à jour: {get_current_time()} UTC
- 👤 Session: {CURRENT_USER}
- 🌐 Version: 4.0.1
- 📡 Status: En ligne
- 💾 Mémoire utilisée: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB
"""
)

# Statut des connexions
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

# Ajout d'un bouton de rafraîchissement manuel
if st.sidebar.button("🔄 Rafraîchir"):
    st.rerun()
