import tensorflow as tf
from packaging import version

# Workaround pour la version Keras
if not hasattr(tf.keras, '__version__'):
    tf.keras.__version__ = tf.__version__

print(f"TensorFlow: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")
print(f"GPU disponibles: {tf.config.list_physical_devices('GPU')}")
