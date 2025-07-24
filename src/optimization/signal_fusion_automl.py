import os
import json
import numpy as np
import pandas as pd
import optuna
import time
import functools

from datetime import datetime
from src.backtesting.core.backtest_engine import BacktestEngine
from src.bot_runner import calculate_position_size
from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators

from binance.client import Client
from dotenv import load_dotenv

from src.bot_runner import TradingBotM4
from src.analysis.news.sentiment_analyzer import NewsSentimentAnalyzer
from src.risk_tools.news_pause_manager import NewsPauseManager

# === INSTANCE DU BOT (NE PAS MODIFIER CETTE LIGNE SANS RAISON) ===
bot = TradingBotM4()

# === INSTANTIATE NEWS ANALYZER AND LOAD NEWS BUFFER ===
bot.news_analyzer = NewsSentimentAnalyzer(bot.config)
try:
    if hasattr(bot.news_analyzer, "fetch_all_news"):
        fetch_result = bot.news_analyzer.fetch_all_news()
        if hasattr(fetch_result, "__await__"):
            import asyncio

            asyncio.run(fetch_result)
except Exception as e:
    print(f"[WARN] Unable to fetch news for news_analyzer: {e}")

# === NEWS PAUSE SYSTEME : INSTANTIATION ===
pause_manager = NewsPauseManager(pause_cycles=5)


def enrich_signals_with_real_values(bot, df, pair_key, news_list=None):
    indics = bot.add_indicators(df)
    rsi = indics.get("rsi_14", 50)
    df["signal_tech"] = (rsi - 50) / 50

    for col, default in [("close", 0.0), ("high", 0.0), ("low", 0.0), ("volume", 0.0)]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    df["rsi"] = df["rsi_14"] if "rsi_14" in df.columns else 50.0
    if "macd" not in df.columns and "macd" in indics:
        df["macd"] = indics["macd"] if indics["macd"] is not None else 0.0
    elif "macd" not in df.columns:
        df["macd"] = 0.0
    df["macd"] = df["macd"].fillna(0.0)
    if "volatility" not in df.columns:
        if "volatility" in indics and indics["volatility"] is not None:
            df["volatility"] = indics["volatility"]
        else:
            df["volatility"] = 0.0
    df["volatility"] = df["volatility"].fillna(0.0)
    df["rsi"] = df["rsi"].fillna(50.0)

    # 2. Signal IA (DeepLearningModel)
    if hasattr(bot, "dl_model") and bot.dl_model:
        from src.ai.deep_learning_model import features_to_array

        def ia_predictor(row):
            idx = row.name
            window = df.loc[:idx].tail(63).copy()
            for col, default in [
                ("close", 0.0),
                ("high", 0.0),
                ("low", 0.0),
                ("volume", 0.0),
                ("rsi", 50.0),
                ("macd", 0.0),
                ("volatility", 0.0),
            ]:
                if col not in window.columns:
                    window[col] = default
                window[col] = window[col].fillna(default)
            if len(window) < 10:
                return 0.0
            features = {
                "close": np.array(window["close"]),
                "high": np.array(window["high"]),
                "low": np.array(window["low"]),
                "volume": np.array(window["volume"]),
                "rsi": np.array(window["rsi"]),
                "macd": np.array(window["macd"]),
                "volatility": np.array(window["volatility"]),
            }
            try:
                return float(bot.dl_model.predict(features))
            except Exception as e:
                print(f"Error in DL prediction: {e}")
                return 0.0

        df["signal_ia"] = df.apply(ia_predictor, axis=1)
    else:
        df["signal_ia"] = 0.0

    # 3. Signal sentiment (NewsSentimentAnalyzer)
    if hasattr(bot, "news_analyzer") and bot.news_analyzer:
        import asyncio

        # On veut toujours utiliser la news_list du cycle courant si disponible !
        if news_list is None:
            news_list = asyncio.run(bot.news_analyzer.fetch_all_news())
        sentiment_score = asyncio.run(
            bot.news_analyzer.get_symbol_sentiment(pair_key, news_list=news_list)
        )
        df["signal_sentiment"] = sentiment_score
    else:
        df["signal_sentiment"] = 0.0

    return df


BEST_PARAMS_PATH = "config/best_signal_params.json"

load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_INTERVAL_MAP = {
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
}


def fetch_binance_ohlcv(
    symbol, interval, start_str, end_str, api_key, api_secret, retries=3, timeout=60
):
    """
    Récupère les données OHLCV depuis Binance avec gestion du timeout et des retries.
    """
    from time import sleep

    client = Client(api_key, api_secret)
    client.session.request = functools.partial(client.session.request, timeout=timeout)
    last_exception = None

    for attempt in range(retries):
        try:
            klines = client.get_historical_klines(symbol, interval, start_str, end_str)
            break
        except Exception as e:
            print(
                f"[FETCH] Attempt {attempt+1}/{retries} failed for {symbol} {interval} (error: {e})"
            )
            last_exception = e
            sleep(5)
    else:
        print(f"[FETCH] All retries failed for {symbol} {interval}")
        raise last_exception

    if not klines or len(klines) == 0:
        print(f"[FETCH] No data for {symbol} {interval}")
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
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fusion_signal_series(df, fusion_params):
    def getval(val):
        if hasattr(val, "__await__"):
            import asyncio

            return asyncio.run(val)
        return val

    def fusion(row):
        score = np.clip(
            fusion_params["tech_weight"] * getval(row.get("signal_tech", 0))
            + fusion_params["ia_weight"] * getval(row.get("signal_ia", 0))
            + fusion_params["sentiment_weight"]
            * getval(row.get("signal_sentiment", 0)),
            -10,
            10,
        )
        if score > fusion_params["buy_threshold"]:
            return 1
        elif score < fusion_params["sell_threshold"]:
            return -1
        return 0

    return df.apply(fusion, axis=1)


def run_full_backtest(df, fusion_params, initial_capital=10000, verbose=False):
    signals = fusion_signal_series(df, fusion_params)
    results = BacktestEngine(initial_capital=initial_capital).run_backtest(
        df, lambda *_args, **_kwargs: signals
    )
    if verbose:
        print(f"[BACKTEST] Résultat: {results}")
    return results


def print_trial_progress(study, trial):
    print(
        f"[Optuna] Trial {trial.number} terminé : Value={trial.value} | "
        f"Params={trial.params} | Best Value={study.best_value:.4f} (Trial {study.best_trial.number})"
    )


def save_best_params(study, trial, path=BEST_PARAMS_PATH):
    best_params = study.best_trial.params
    best_params["score"] = study.best_value
    with open(path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"[Optuna] Best params sauvegardés: {best_params}")


def optuna_callback(study, trial):
    print_trial_progress(study, trial)
    save_best_params(study, trial)


def objective(trial):
    # Optionally, fetch news at the start of each trial to refresh buffer
    if hasattr(bot, "news_analyzer"):
        fetch_result = bot.news_analyzer.fetch_all_news()
        if hasattr(fetch_result, "__await__"):
            import asyncio

            asyncio.run(fetch_result)

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

            # === NEWS PAUSE SYSTEME : ICI ===
            import asyncio

            news_list = asyncio.run(bot.news_analyzer.fetch_all_news())
            if pause_manager.scan_news(news_list):
                print("🚨 Pause trading à cause d'une news critique !")
            if pause_manager.should_pause():
                print("Trading en pause, on skip ce cycle.")
                pause_manager.on_cycle_end()
                continue

            df = enrich_signals_with_real_values(
                bot, df, pair_key=pair.replace("/", ""), news_list=news_list
            )

            results = run_full_backtest(df, fusion_params, initial_capital=10000)
            profit = results.get("final_capital", 0) - 10000 if results else -9999
            if profit is None or np.isnan(profit):
                profit = -99999
            all_scores.append(profit)
            time.sleep(1)  # Limite la fréquence des appels API
            pause_manager.on_cycle_end()

    avg_profit = np.mean(all_scores) if all_scores else -99999
    print(f"[OPTUNA] Params: {fusion_params} | Score: {avg_profit:.2f}")
    return avg_profit


def optimize_signal_fusion_and_mm(n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
    print("Best params:", study.best_params)
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params


if __name__ == "__main__":
    print("=== OPTIMISATION SIGNAL FUSION & MM ===")
    best = optimize_signal_fusion_and_mm(n_trials=100)
    print("Meilleure configuration trouvée :", best)
