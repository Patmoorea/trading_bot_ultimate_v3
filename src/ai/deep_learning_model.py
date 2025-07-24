import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import os


def features_to_array(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Transforme un dict de features (avec arrays ou scalaires) en un unique array 2D
    adapté à l'entrée du modèle CNN-LSTM.
    - close, high, low, volume, rsi, macd, volatility : array shape (N,)
    Retourne un array shape (N, 7)
    """
    close = np.array(features["close"])
    high = np.array(features["high"])
    low = np.array(features["low"])
    volume = np.array(features["volume"])
    N = close.shape[0]
    rsi = np.array(features["rsi"])
    if np.isscalar(rsi) or rsi.shape == ():
        rsi = np.full(N, rsi)
    macd = np.array(features["macd"])
    if np.isscalar(macd) or macd.shape == ():
        macd = np.full(N, macd)
    volatility = np.array(features["volatility"])
    if np.isscalar(volatility) or volatility.shape == ():
        volatility = np.full(N, volatility)
    arr = np.stack([close, high, low, volume, rsi, macd, volatility], axis=1)
    return arr


class DeepLearningModel:
    def __init__(self):
        self.model = CNNLSTMModel()
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            path = "models/cnn_lstm_model.pth"
            if os.path.exists(path):
                try:
                    print(f"[DEBUG] torch.load: path={path}")
                    state_dict = torch.load(path, map_location=torch.device("cpu"))
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    print(f"[DL] Modèle chargé depuis {path}")
                except Exception as e:
                    print(f"[WARN] Erreur chargement modèle: {e}")
            else:
                print(f"[WARN] Aucun modèle entraîné trouvé à {path}")
            self.initialized = True

    def predict(self, features: Dict[str, np.ndarray]) -> float:
        """
        Prédit la sortie du modèle.
        ATTENTION: `features` doit toujours être un dictionnaire
        avec les clés 'close', 'high', 'low', 'volume', 'rsi', 'macd', 'volatility'
        et chaque entrée doit être un np.array shape (N,) ou un scalaire.
        """
        try:
            if not isinstance(features, dict) or any(
                k not in features
                for k in ["close", "high", "low", "volume", "rsi", "macd", "volatility"]
            ):
                raise ValueError(
                    "Features must be a dict with keys: close, high, low, volume, rsi, macd, volatility"
                )
            if features["close"].shape[0] < 10:
                return 0.0
            if not self.initialized:
                self.initialize()
            x = self._prepare_features(features)
            with torch.no_grad():
                prediction = self.model(x)
            return float(prediction.item())
        except Exception as e:
            print(f"Error in DL prediction: {e}")
            return np.random.uniform(0.0, 0.1)

    def _prepare_features(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        arr = features_to_array(features)  # shape (N, 7)
        arr = arr.T  # (7, N)
        arr = np.expand_dims(arr, axis=0)  # (1, 7, N)
        return torch.FloatTensor(arr)

    def load_weights(self, path):
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path, map_location="cpu"))
                print(f"[DEBUG] torch.load: path={path}")
                self.model.eval()
                print(f"[DL] Modèle chargé depuis {path}")
            except Exception as e:
                print(f"[WARN] Erreur chargement modèle: {e}")
        else:
            print(f"[WARN] Fichier modèle {path} absent, chargement ignoré.")


class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, channels=7, seq_len)
        x = self.cnn(x)  # (batch, 64, seq_len')
        x = x.permute(0, 2, 1)  # (batch, seq_len', 64) pour LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Prendre la dernière sortie (batch, 128)
        x = self.fc(x)  # (batch, 1)
        return x
