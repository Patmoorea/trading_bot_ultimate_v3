import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from src.ai.deep_learning_model import CNNLSTMModel

SEQ_LEN = 20
FUTURE_SHIFT = 10
THRESHOLD = 0.002


def load_data_from_df(df):
    for col in ["close", "high", "low", "volume", "rsi", "macd", "volatility"]:
        if col in df:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    X, y = [], []
    for i in range(len(df) - SEQ_LEN - FUTURE_SHIFT):
        features = []
        for col in ["close", "high", "low", "volume", "rsi", "macd", "volatility"]:
            if col in df:
                features.append(df[col].iloc[i : i + SEQ_LEN].values)
        feat_arr = np.stack(features, axis=1)
        X.append(feat_arr)
        future_close = df["close"].iloc[i + SEQ_LEN + FUTURE_SHIFT - 1]
        now_close = df["close"].iloc[i + SEQ_LEN - 1]
        label = 1 if (future_close - now_close) / abs(now_close) > THRESHOLD else 0
        y.append(label)
    if not X or not y:
        print("Aucune donnée d'entraînement disponible")
        return None, None
    X = np.stack(X)
    y = np.array(y).reshape(-1, 1)
    return X, y


def train_with_live_data(df_live, model_save_path="src/models/cnn_lstm_model.pth"):
    X, y = load_data_from_df(df_live)
    if X is None or y is None:
        print("Pas assez de données pour entraîner le modèle.")
        return
    print("Features shape:", X.shape, "Targets shape:", y.shape)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=42
    )
    model = CNNLSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    n_epochs = 20
    batch_size = 64
    for epoch in range(n_epochs):
        model.train()
        idxs = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idxs], y_train[idxs]
        batch_losses = []
        for i in range(0, len(X_train), batch_size):
            xb = torch.FloatTensor(X_train[i : i + batch_size]).transpose(1, 2)
            yb = torch.FloatTensor(y_train[i : i + batch_size])
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            xb = torch.FloatTensor(X_val).transpose(1, 2)
            yb = torch.FloatTensor(y_val)
            y_pred = model(xb)
            val_loss = loss_fn(y_pred, yb).item()
            acc = ((y_pred > 0.5).float() == yb).float().mean().item()
        print(
            f"Epoch {epoch+1}/{n_epochs} - Train Loss: {np.mean(batch_losses):.4f}  Val Loss: {val_loss:.4f}  Val Acc: {acc:.3f}"
        )
    import os

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Modèle entraîné et sauvegardé à {model_save_path}")
