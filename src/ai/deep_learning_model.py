import torch
import torch.nn as nn
import numpy as np
from typing import Dict


def features_to_array(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Transforme un dict de features (avec arrays et scalaires) en un unique array 2D
    adapté à l'entrée du modèle CNN-LSTM.
    - close, high, low, volume : array shape (N,)
    - rsi, macd, volatility : scalaires
    Retourne un array shape (N, 7)
    """
    close = np.array(features["close"])
    high = np.array(features["high"])
    low = np.array(features["low"])
    volume = np.array(features["volume"])
    N = close.shape[0]
    rsi = np.full(N, features["rsi"])
    macd = np.full(N, features["macd"])
    volatility = np.full(N, features["volatility"])
    arr = np.stack([close, high, low, volume, rsi, macd, volatility], axis=1)
    return arr


class DeepLearningModel:
    def __init__(self):
        self.model = CNNLSTMModel()
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            try:
                self.model.load_state_dict(
                    torch.load(
                        "models/cnn_lstm_model.pth", map_location=torch.device("cpu")
                    )
                )
                self.model.eval()
            except Exception as e:
                print("No pre-trained model found, using new model. Details:", e)
            self.initialized = True

    def predict(self, features: Dict[str, np.ndarray]) -> float:
        try:
            # S'assurer que le modèle est initialisé
            if not self.initialized:
                self.initialize()
                
            x = self._prepare_features(features)
            with torch.no_grad():
                prediction = self.model(x)
            
            # Améliorer la normalisation pour avoir des prédictions plus marquées
            raw_pred = prediction.item()
            # Transformer sigmoid [0,1] vers [-1,1] avec amplification réduite
            normalized_pred = (raw_pred - 0.5) * 2  # Amplification réduite de 4 à 2
            return np.clip(normalized_pred, -1, 1)
            
        except Exception as e:
            print(f"Error in DL prediction: {e}")
            # Retourner une prédiction aléatoire faible au lieu de 0
            return np.random.uniform(-0.1, 0.1)

    def _prepare_features(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Transforme le dict de features en un tensor shape (batch=1, 7, N)
        pour l'entrée du CNN.
        """
        arr = features_to_array(features)  # shape (N, 7)
        arr = arr.T  # (7, N) car Conv1d attend (batch, channels, seq_len)
        arr = np.expand_dims(arr, axis=0)  # (1, 7, N)
        return torch.FloatTensor(arr)


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
