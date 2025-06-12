import time
from functools import lru_cache


@lru_cache(maxsize=32)
def cached_news(keywords, max_age=3600):
    """Cache les résultats pour 1 heure"""
    return EnhancedNewsProcessor().get_crypto_news(keywords)


def configure_quantum():
    """Active les optimisations Apple Silicon"""
    import os

    os.environ["PYLON_RUN"] = "metal"  # Force le backend Metal
    os.environ["OMP_NUM_THREADS"] = "8"  # Utilise les 8 cores


def track_latency():
    """Nouveau monitoring temps-réel"""
    return {"decision": avg_time_ms, "execution": exec_time_ms}
