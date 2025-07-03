import torch
from typing import Dict, Any
from stable_baselines3 import PPO
from torch.nn import TransformerEncoderLayer
from ..strategies.base import BaseStrategy
from src.ai_decision.ppo_transformer import PPOTradingAgent
import requests


class PPOStrategy(BaseStrategy):
    """
    Stratégie de trading utilisant PPO avec Transformer
    Réutilise le code existant de PPOTradingAgent
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.agent = PPOTradingAgent(
            env=config["env"]  # Votre environnement de trading existant
        )

    def initialize(self):
        """Initialisation avec le modèle pré-entraîné si disponible"""
        if "model_path" in self.config:
            self.agent.model = PPO.load(self.config["model_path"])

    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Réutilise la logique de décision de PPOTradingAgent"""
        try:
            # Utilise le même format que votre PPOModel existant
            decision = await self.agent.model.predict(market_data, deterministic=True)

            return {
                "action": self._convert_action(decision[0]),
                "confidence": float(decision[1].max()),
                "size": self.config.get("position_size", 1000),
            }

        except Exception as e:
            self.logger.error(f"Erreur dans PPOStrategy: {e}")
            return None

    def _convert_action(self, action: int) -> str:
        """Conversion des actions numériques en décisions de trading"""
        # Utilise le même mapping que dans votre code existant
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, "HOLD")

    def cleanup(self):
        """Sauvegarde le modèle si nécessaire"""
        if "save_path" in self.config:
            self.agent.model.save(self.config["save_path"])

    def fetch_cointelegraph_news():
        url = "https://cointelegraph.com/api/v1/news"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith(
            "application/json"
        ):
            return resp.json()
        else:
            print(
                f"Erreur Cointelegraph: code={resp.status_code}, content-type={resp.headers.get('Content-Type')}"
            )
            return []
