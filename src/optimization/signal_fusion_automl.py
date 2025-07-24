import os
import json
import numpy as np
import pandas as pd
import optuna
from datetime import datetime
from src.backtesting.core.backtest_engine import BacktestEngine
from src.bot_runner import calculate_position_size
from src.analysis.technical.advanced.advanced_indicators import AdvancedIndicators

# Chemins pour sauvegarde des meilleurs params
BEST_PARAMS_PATH = "config/best_signal_params.json"


def load_historical_data(pair, timeframe, data_dir="data/historical"):
    """Charge les données historiques au format CSV."""
    fname = f"{pair.replace('/', '')}_{timeframe}.csv"
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        print(f"[DATA] Fichier historique manquant: {path}")
        return None
    df = pd.read_csv(path)
    return df


def run_full_backtest(
    df,
    fusion_params,
    strategy_func,
    initial_capital=10000,
    verbose=False,
):
    """
    Lance un backtest sur un DataFrame, avec pondérations personnalisées.
    fusion_params: dict, doit contenir les poids pour tech/ia/sentiment, seuils, etc.
    """
    # Exemple de passage de params à la stratégie
    # La stratégie doit accepter **fusion_params (adapter selon ton implémentation)
    results = BacktestEngine(initial_capital=initial_capital).run_backtest(
        df, strategy_func, fusion_params=fusion_params
    )
    if verbose:
        print(f"[BACKTEST] Résultat: {results}")
    return results


def objective(trial):
    """
    Fonction objective Optuna pour optimiser les pondérations et seuils.
    """
    # 1. Définir les hyperparams à tuner
    tech_weight = trial.suggest_float("tech_weight", 0.0, 1.0)
    ia_weight = trial.suggest_float("ia_weight", 0.0, 1.0 - tech_weight)
    sentiment_weight = 1.0 - tech_weight - ia_weight

    buy_threshold = trial.suggest_float("buy_threshold", 0.1, 0.5)
    sell_threshold = trial.suggest_float("sell_threshold", -0.5, -0.1)
    mm_risk = trial.suggest_float(
        "mm_risk", 0.01, 0.2
    )  # Money management: % du capital par trade

    # 2. Fusion params
    fusion_params = {
        "tech_weight": tech_weight,
        "ia_weight": ia_weight,
        "sentiment_weight": sentiment_weight,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "mm_risk": mm_risk,
    }

    # 3. Sélection des paires/timeframes pour l'optimisation
    pairs = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
    timeframes = ["1h", "4h"]
    all_scores = []
    for pair in pairs:
        for tf in timeframes:
            df = load_historical_data(pair, tf)
            if df is None or len(df) < 100:
                continue
            # STRAT adaptative à ta logique (doit accepter fusion_params)
            results = run_full_backtest(
                df,
                fusion_params,
                strategy_func=simple_fusion_strategy,
                initial_capital=10000,
            )
            # Score = profit final, tu peux pondérer avec Sharpe, drawdown, etc.
            profit = results.get("final_balance", 0) - 10000
            all_scores.append(profit)
    # Moyenne sur tous les tests
    avg_profit = np.mean(all_scores) if all_scores else -99999
    return avg_profit


def simple_fusion_strategy(row, fusion_params):
    # row: dict ou pd.Series avec les signaux calculés comme dans ton bot
    score = (
        fusion_params["tech_weight"] * row.get("signal_tech", 0)
        + fusion_params["ia_weight"] * row.get("signal_ia", 0)
        + fusion_params["sentiment_weight"] * row.get("signal_sentiment", 0)
    )
    if score > fusion_params["buy_threshold"]:
        return "buy"
    elif score < fusion_params["sell_threshold"]:
        return "sell"
    return "hold"


def optimize_signal_fusion_and_mm(n_trials=50):
    """
    Fonction principale pour lancer l'optimisation globale.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params


if __name__ == "__main__":
    print("=== OPTIMISATION SIGNAL FUSION & MM ===")
    best = optimize_signal_fusion_and_mm(n_trials=100)
    print("Meilleure configuration trouvée :", best)
