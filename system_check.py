import sys
import numpy as np

print("=== Diagnostic Système ===")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")

try:
    import tensorflow as tf
    print(f"\nTensorFlow: {tf.__version__}")
    print(f"GPU détectés: {tf.config.list_physical_devices('GPU')}")
except Exception as e:
    print(f"\nERREUR TensorFlow: {e}")

try:
    import jax
    print(f"\nJAX: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
except Exception as e:
    print(f"\nJAX non disponible: {e}")

print("\nTest complet TensorFlow:")
try:
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    print("✓ Modèle Keras créé avec succès")
except Exception as e:
    print(f"✗ Erreur modèle: {e}")
