import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from src.ai.deep_learning_model import CNNLSTMModel
import json

SEQ_LEN = 20
FUTURE_SHIFT = 10
THRESHOLD = 0.002

def load_best_params(path="config/best_hyperparams.json"):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_checkpoint(model, optimizer, epoch, path="src/models/checkpoint.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, path="src/models/checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def load_data_from_df(df, seq_len=20, future_shift=10, threshold=0.002):
    """
    Prépare les features et labels pour l'entraînement IA.
    - Normalise chaque colonne.
    - Gère les NaN/inf.
    - Génère X (features) et y (labels binaires).
    - Affiche la distribution des labels pour debug.
    """

    feature_cols = ["close", "high", "low", "volume", "rsi", "macd", "volatility"]
    for col in feature_cols:
        if col not in df:
            print(f"Manque la colonne {col} dans le DataFrame, impossible d’entraîner !")
            return None, None
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        if df[col].isnull().any() or np.isinf(df[col]).any():
            print(f"Colonne {col} contient nan ou inf après normalisation !")
    df = df.dropna(subset=feature_cols)
    X, y = [], []
    for i in range(len(df) - seq_len - future_shift):
        features = [df[col].iloc[i : i + seq_len].values for col in feature_cols]
        if any(np.isnan(f).any() or np.isinf(f).any() for f in features):
            continue  # Skip si NaN/inf dans la fenêtre
        feat_arr = np.stack(features, axis=1)
        X.append(feat_arr)
        future_close = df["close"].iloc[i + seq_len + future_shift - 1]
        now_close = df["close"].iloc[i + seq_len - 1]
        # Calcul du label binaire
        try:
            label = 1.0 if (future_close - now_close) / abs(now_close) > threshold else 0.0
        except Exception:
            label = 0.0
        y.append(label)
    if not X or not y:
        print("Aucune donnée d'entraînement disponible")
        return None, None
    X = np.stack(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    # Debug: distribution des labels
    print(f"[DEBUG] Nombre de labels 1 : {np.sum(y == 1)}")
    print(f"[DEBUG] Nombre de labels 0 : {np.sum(y == 0)}")
    print(f"[DEBUG] Proportion de labels 1 : {np.mean(y):.4f}")
    print("DEBUG X min:", np.nanmin(X), "X max:", np.nanmax(X))
    print("DEBUG X has nan:", np.isnan(X).any(), "X has inf:", np.isinf(X).any())
    print("DEBUG y min:", y.min(), "y max:", y.max(), "dtype:", y.dtype)
    return X, y

def add_dl_features(df):
    import pandas_ta as pta
    # RSI 14
    if "rsi" not in df:
        df["rsi"] = pta.rsi(df["close"], length=14)
    # MACD (on prend la ligne MACD)
    if "macd" not in df:
        macd = pta.macd(df["close"])
        df["macd"] = macd["MACD_12_26_9"] if "MACD_12_26_9" in macd else np.nan
    # Volatility (écart-type des returns)
    if "volatility" not in df:
        returns = np.log(df["close"]).diff()
        df["volatility"] = returns.rolling(14).std()
    return df

def train_with_live_data(df_live, model_save_path="src/models/cnn_lstm_model.pth"):
    # Charge les meilleurs hyperparams issus d'Optuna/AutoML
    best_params = load_best_params()
    lr = best_params.get("lr", 0.001)
    n_epochs = best_params.get("n_epochs", 100)
    batch_size = best_params.get("batch_size", 64)

    X, y = load_data_from_df(df_live)
    if X is None or y is None:
        print("Pas assez de données pour entraîner le modèle.")
        return
    print("Features shape:", X.shape, "Targets shape:", y.shape)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=42
    )
    model = CNNLSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    checkpoint_path = "src/models/checkpoint.pth"
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    if start_epoch > 0:
        print(f"✅ Reprise de l'entraînement à l'epoch {start_epoch+1}")
    else:
        print("⏩ Entraînement à partir de zéro.")

    for epoch in range(start_epoch, n_epochs):
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
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {np.mean(batch_losses):.6f}")

        # Sauvegarde du checkpoint à chaque epoch
        save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
        
        model.eval()
        with torch.no_grad():
            xb = torch.FloatTensor(X_val).transpose(1, 2)
            yb = torch.FloatTensor(y_val)
            y_pred = model(xb)
            val_loss = loss_fn(y_pred, yb).item()
            acc = ((y_pred > 0.5).float() == yb).float().mean().item()

    # Sauvegarde finale du modèle entraîné
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Modèle entraîné et sauvegardé à {model_save_path}")