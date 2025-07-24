import os
import json
import numpy as np
import pandas as pd
import optuna
from datetime import datetime
from src.backtesting.core.backtest_engine import BacktestEngine
from src.bot_runner import calculate_position_size
from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators

# Ajoute ces imports pour le fetch dynamique
from binance.client import Client
from dotenv import load_dotenv

BEST_PARAMS_PATH = "config/best_signal_params.json"

# Charger les clés API depuis .env
load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_INTERVAL_MAP = {
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    # Ajoute d'autres timeframes si besoin
}


def fetch_binance_ohlcv(symbol, interval, start_str, end_str, api_key, api_secret):
    """Télécharge des données historiques OHLCV depuis Binance."""
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    if not klines or len(klines) == 0:
        print(f"[FETCH] Aucune donnée récupérée pour {symbol} {interval}")
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


def simple_fusion_strategy(row, fusion_params):
    """
    Exemple de stratégie de fusion. Adapter à ton format réel de signaux.
    """
    get = lambda x: row[x] if x in row else row.get(x, 0)
    score = (
        fusion_params["tech_weight"] * get("signal_tech")
        + fusion_params["ia_weight"] * get("signal_ia")
        + fusion_params["sentiment_weight"] * get("signal_sentiment")
    )
    if score > fusion_params["buy_threshold"]:
        return "buy"
    elif score < fusion_params["sell_threshold"]:
        return "sell"
    return "hold"


def run_full_backtest(
    df, fusion_params, strategy_func, initial_capital=10000, verbose=False
):
    results = BacktestEngine(initial_capital=initial_capital).run_backtest(
        df, strategy_func, fusion_params=fusion_params
    )
    if verbose:
        print(f"[BACKTEST] Résultat: {results}")
    return results


def objective(trial):
    tech_weight = trial.suggest_float("tech_weight", 0.0, 1.0)
    ia_weight = trial.suggest_float("ia_weight", 0.0, 1.0 - tech_weight)
    sentiment_weight = 1.0 - tech_weight - ia_weight
    buy_threshold = trial.suggest_float("buy_threshold", 0.1, 0.5)
    sell_threshold = trial.suggest_float("sell_threshold", -0.5, -0.1)
    mm_risk = trial.suggest_float("mm_risk", 0.01, 0.2)

    fusion_params = {
        "tech_weight": tech_weight,
        "ia_weight": ia_weight,
        "sentiment_weight": sentiment_weight,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "mm_risk": mm_risk,
    }

    pairs = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
    timeframes = ["1h", "4h"]
    all_scores = []
    for pair in pairs:
        for tf in timeframes:
            df = fetch_binance_ohlcv(
                symbol=pair.replace("/", ""),
                interval=BINANCE_INTERVAL_MAP[tf],
                start_str="1 Jan, 2023",
                end_str="now",
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET,
            )
            if df is None or len(df) < 100:
                continue
            results = run_full_backtest(
                df,
                fusion_params,
                strategy_func=simple_fusion_strategy,
                initial_capital=10000,
            )
            profit = results.get("final_balance", 0) - 10000 if results else -9999
            all_scores.append(profit)
    avg_profit = np.mean(all_scores) if all_scores else -99999
    print(f"[OPTUNA] Params: {fusion_params} | Score: {avg_profit:.2f}")
    return avg_profit


def optimize_signal_fusion_and_mm(n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params


if __name__ == "__main__":
    print("=== OPTIMISATION SIGNAL FUSION & MM ===")
    best = optimize_signal_fusion_and_mm(n_trials=100)
    print("Meilleure configuration trouvée :", best)
