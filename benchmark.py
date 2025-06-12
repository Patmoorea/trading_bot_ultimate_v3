import tensorflow as tf
import time

print("=== Test TensorFlow ===")
print(f"Version: {tf.__version__}")
print(f"GPU Disponible: {tf.config.list_physical_devices('GPU')}")

size = 10000
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device):
    start = time.time()
    a = tf.random.normal((size, size))
    b = tf.random.normal((size, size))
    c = tf.matmul(a, b)
    print(f"\nTemps calcul matrice {size}x{size} sur {device}: {time.time()-start:.2f}s")
