# Configuration TensorFlow
TF_CONFIG = {
    "GPU_MEMORY_GROWTH": True,
    "XLA_OPTIMIZATIONS": True,
    "MIXED_PRECISION": False  # Désactivé pour la stabilité
}

# Configuration du Bot
BOT_CONFIG = {
    "MAX_DRAWDOWN": 0.05,
    "RISK_PER_TRADE": 0.02,
    "GPU_ACCELERATION": True
}

# Optimisation Apple Silicon (M4)
GPU_CONFIG = {
    'use_metal': True,
    'memory_limit': 0.9,  # 90% pour le M4 plus performant
    'precision': 'mixed_float16'  # Acceleration spécifique M4
}
