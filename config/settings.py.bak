import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Binance API
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    
    # Trading Parameters
    RISK_PERCENT = float(os.getenv('RISK_PERCENT', 2.0))
    MAX_SLIPPAGE = float(os.getenv('MAX_SLIPPAGE', 0.001))
    
    # Model Paths
    MODEL_DIR = "models/"
    CNN_LSTM_WEIGHTS = os.path.join(MODEL_DIR, "cnn_lstm_weights.h5")
    PPO_MODEL = os.path.join(MODEL_DIR, "ppo_model.zip")
