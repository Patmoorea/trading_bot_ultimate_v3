import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, Union
from stable_baselines3 import PPO
from ..strategies.base import BaseStrategy


class PPOStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # Vérification de l'environnement
        self.env = config.get("env")
        if self.env is None:
            self.logger.error("Environnement manquant dans la configuration")
            raise ValueError("Environnement manquant dans la configuration")

        # Vérification des dimensions
        if not hasattr(self.env, "observation_space") or not hasattr(
            self.env, "action_space"
        ):
            self.logger.error(
                "Environnement mal configuré (observation_space ou action_space manquant)"
            )
            raise ValueError("Environnement mal configuré")

        # Configuration PPO
        self.ppo_config = {
            "policy": "MlpPolicy",
            "learning_rate": config.get("learning_rate", 3e-4),
            "n_steps": config.get("n_steps", 2048),
            "batch_size": config.get("batch_size", 64),
            "n_epochs": config.get("n_epochs", 10),
            "gamma": config.get("gamma", 0.99),
            "gae_lambda": config.get("gae_lambda", 0.95),
            "clip_range": config.get("clip_range", 0.2),
            "verbose": config.get("verbose", 1),
            "policy_kwargs": {"net_arch": [64, 64], "activation_fn": torch.nn.ReLU},
        }

        try:
            self.model = PPO(env=self.env, **self.ppo_config)
            self.logger.info("✅ Modèle PPO initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation PPO: {str(e)}")
            self.model = None

        # Vérification des dimensions
        if not hasattr(self.env, "observation_space") or not hasattr(
            self.env, "action_space"
        ):
            self.logger.error(
                "Environnement mal configuré (observation_space ou action_space manquant)"
            )
            return
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        # Vérification de l'environnement
        self.env = config.get("env")
        if self.env is None:
            raise ValueError("Environnement manquant dans la configuration")

        # Dimensions
        self.input_dim = config.get("input_dim", 42)
        self.action_dim = len(self.env.trading_pairs)

        # Configuration PPO avec valeurs par défaut
        self.ppo_config = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 1,
            "policy_kwargs": {"net_arch": [64, 64], "activation_fn": torch.nn.ReLU},
        }

        # Initialisation explicite du modèle
        try:
            self.model = PPO(env=self.env, **self.ppo_config)
            self.logger.info("✅ Modèle PPO initialisé")
        except Exception as e:
            self.logger.error(f"❌ Erreur initialisation PPO: {str(e)}")
            raise  # Relever l'erreur pour voir la pile complète

    def get_action(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Méthode explicite get_action requise par le bot
        """
        try:
            if self.model is None:
                return {"action": "HOLD", "confidence": 0.0}

            # Préparation de l'état
            processed_state = self._preprocess_state(state)

            # Prédiction
            action, _ = self.model.predict(processed_state, deterministic=True)

            # Calcul de la confiance
            confidence = self._calculate_confidence(processed_state)

            return {
                "action": self._convert_action(action),
                "confidence": confidence,
                "raw_action": action,
            }

        except Exception as e:
            self.logger.error(f"Erreur get_action: {str(e)}")
            return {"action": "HOLD", "confidence": 0.0}

    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """Prétraitement de l'état"""
        try:
            # Conversion en numpy si nécessaire
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=np.float32)

            # Reshape si nécessaire
            if state.ndim == 1:
                state = state.reshape(1, -1)

            # Padding/Truncating pour avoir la bonne dimension
            target_shape = (1, self.input_dim)
            if state.shape != target_shape:
                if state.shape[1] > self.input_dim:
                    state = state[:, : self.input_dim]
                else:
                    pad_width = ((0, 0), (0, self.input_dim - state.shape[1]))
                    state = np.pad(state, pad_width, mode="constant")

            return state

        except Exception as e:
            self.logger.error(f"Erreur prétraitement: {str(e)}")
            return np.zeros((1, self.input_dim))

    def _calculate_confidence(self, state: np.ndarray) -> float:
        """Calcul du score de confiance"""
        try:
            if self.model is None:
                return 0.0

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                dist = self.model.policy.get_distribution(state_tensor)
                probs = dist.distribution.probs
                return float(torch.max(probs))

        except Exception as e:
            self.logger.error(f"Erreur calcul confiance: {str(e)}")
            return 0.0

    def _convert_action(self, action: Union[int, np.ndarray]) -> str:
        """Conversion action -> décision"""
        try:
            if isinstance(action, np.ndarray):
                action = action.item()

            actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
            return actions.get(int(action), "HOLD")

        except Exception as e:
            self.logger.error(f"Erreur conversion action: {str(e)}")
            return "HOLD"

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse du marché"""
        try:
            state = self._prepare_observation(market_data)
            action_dict = self.get_action(state)

            return {
                "action": action_dict["action"],
                "confidence": action_dict["confidence"],
                "size": self.config.get("position_size", 1000),
                "metadata": {
                    "raw_action": action_dict["raw_action"],
                    "timestamp": market_data.get("timestamp"),
                },
            }

        except Exception as e:
            self.logger.error(f"Erreur analyze_market: {str(e)}")
            return {"action": "HOLD", "confidence": 0.0}

    def _prepare_observation(self, market_data: Dict[str, Any]) -> np.ndarray:
        try:
            features = []

            # Log des dimensions pour déboguer
            for key, value in market_data.items():
                self.logger.debug(
                    f"Dimension de {key}: {np.array(value).shape if isinstance(value, (list, np.ndarray)) else 'scalaire'}"
                )

            # OHLCV
            if "ohlcv" in market_data:
                ohlcv = np.array(market_data["ohlcv"], dtype=np.float32)
                if len(ohlcv) > 0:
                    features.append(ohlcv[-1])  # Dernier état

            # Indicateurs
            if "indicators" in market_data:
                indicators = np.array(
                    list(market_data["indicators"].values()), dtype=np.float32
                )
                features.append(indicators)

            # Métriques
            if "market_metrics" in market_data:
                metrics = np.array(
                    list(market_data["market_metrics"].values()), dtype=np.float32
                )
                features.append(metrics)

            if not features:
                return np.zeros(self.input_dim)

            # Combinaison des features
            state = np.concatenate(features).flatten()
            return self._preprocess_state(state)

        except Exception as e:
            self.logger.error(f"Erreur prepare_observation: {str(e)}")
            return np.zeros(self.input_dim)
