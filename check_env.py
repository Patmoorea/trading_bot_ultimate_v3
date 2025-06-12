import sys
import numpy as np
import tensorflow as tf

print("=== Configuration Valide ===")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Disponible: {tf.config.list_physical_devices('GPU')}")
