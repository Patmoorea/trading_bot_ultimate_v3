import optuna


def tune_hyperparameters():
    """Optimisation des hyperparamètres avec Optuna"""
    # Import retardé pour éviter les dépendances circulaires
    from src.core_merged.ai_engine import HybridAI

    def objective(trial):
        model = HybridAI()
        model.learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        return model.validate() if hasattr(model, 'validate') else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params


def optimize_hyperparameters_full():
    """Optimisation complète CNN-LSTM + Transformer"""
    study = optuna.create_study(directions=['maximize', 'minimize'])

    def objective(trial):
        model = HybridAIEnhanced()
        model.learning_rate = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
        model.dropout = trial.suggest_float('dropout', 0.1, 0.5)

        # Score composite
        accuracy = model.validate()
        latency = model.get_latency()
        return accuracy, latency

    study.optimize(objective, n_trials=200, timeout=3600)
    return study.best_trials
