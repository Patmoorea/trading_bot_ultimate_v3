import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class DeepLearningModel:
    def __init__(self):
        self.model = CNNLSTMModel()
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            try:
                self.model.load_state_dict(torch.load("models/cnn_lstm_model.pth"))
                self.model.eval()
            except:
                print("No pre-trained model found, using new model")
            self.initialized = True

    def predict(self, features: Dict[str, np.ndarray]) -> float:
        try:
            # Préparation des données
            x = self._prepare_features(features)
            
            # Prédiction
            with torch.no_grad():
                prediction = self.model(x)
                
            # Normalisation entre -1 et 1
            return prediction.item() * 2 - 1
            
        except Exception as e:
            print(f"Error in DL prediction: {e}")
            return 0.0

    def _prepare_features(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        feature_array = []
        for key in ['close', 'high', 'low', 'volume', 'rsi', 'macd', 'volatility']:
            if key in features:
                feature_array.append(features[key])
        
        x = np.stack(feature_array, axis=1)
        return torch.FloatTensor(x).unsqueeze(0)

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CNN
        x = self.cnn(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # Réorganiser pour LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Prendre la dernière sortie
        
        # FC
        x = self.fc(x)
        return x
