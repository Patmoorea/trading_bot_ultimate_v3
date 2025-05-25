import numpy as np
from hmmlearn import hmm
from sklearn.cluster import KMeans

class MarketRegimeDetector:
    def __init__(self, n_regimes=3):  # Réduit à 3 régimes par défaut
        self.hmm = hmm.GaussianHMM(n_components=n_regimes)
        self.kmeans = KMeans(n_clusters=n_regimes)
        self.regimes = {
            0: "Bull",
            1: "Bear", 
            2: "Sideways"
        }

    def fit(self, prices: np.ndarray):
        prices = np.array(prices, dtype=np.float64)
        if len(prices) < 10:  # Minimum 10 points de données
            raise ValueError("Au moins 10 prix sont nécessaires pour l'entraînement")
        
        returns = np.log(prices[1:]/prices[:-1])
        self.hmm.fit(returns.reshape(-1, 1))
        self.kmeans.fit(returns.reshape(-1, 1))

    def predict(self, window: np.ndarray) -> str:
        window = np.array(window, dtype=np.float64)
        if len(window) < 5:  # Minimum 5 points pour la prédiction
            return "Insufficient Data"
            
        returns = np.log(window[1:]/window[:-1])
        regime_id = self.hmm.predict(returns.reshape(-1, 1))[-1]
        return self.regimes[regime_id]

class OptimizedMarketRegimeDetector(MarketRegimeDetector):
    """Version avec paramètres optimisés pour la convergence"""
    def __init__(self, n_regimes=3):
        super().__init__(n_regimes)
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            tol=1e-4,
            n_iter=1000,
            init_params='ste',  # Initialisation plus stable
            verbose=True
        )

class OptimizedMarketRegimeDetector:
    """Version améliorée - COEXISTE avec MarketRegimeDetector"""
    def __init__(self, n_regimes=3):
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            tol=1e-4,
            n_iter=1000,
            init_params='ste',
            verbose=False
        )
        self.kmeans = KMeans(n_clusters=n_regimes)
        self.regimes = {
            0: "Bull",
            1: "Bear", 
            2: "Sideways"
        }

    def fit(self, prices: np.ndarray):
        prices = np.array(prices, dtype=np.float64)
        if len(prices) < 10:
            raise ValueError("Requiert au moins 10 points de données")
        returns = np.log(prices[1:]/prices[:-1])
        self.hmm.fit(returns.reshape(-1, 1))
        self.kmeans.fit(returns.reshape(-1, 1))

    def predict(self, window: np.ndarray) -> str:
        window = np.array(window, dtype=np.float64)
        if len(window) < 5:
            return "Insufficient Data"
        returns = np.log(window[1:]/window[:-1])
        regime_id = self.hmm.predict(returns.reshape(-1, 1))[-1]
        return self.regimes[regime_id]
