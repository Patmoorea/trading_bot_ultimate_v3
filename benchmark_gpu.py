import tensorflow as tf
import time

size = 10000
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device):
    start = time.time()
    a = tf.random.normal((size, size))
    b = tf.random.normal((size, size))
    c = tf.matmul(a, b)
    duration = time.time() - start

print(f"Calcul matrice {size}x{size} sur {device}: {duration:.2f}s")
