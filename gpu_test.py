import tensorflow as tf
import time

# Configuration
size = 5000
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Test
with tf.device(device):
    start = time.time()
    a = tf.random.normal((size, size))
    b = tf.random.normal((size, size))
    c = tf.matmul(a, b)
    print(f"\nTemps de calcul sur {device}: {time.time()-start:.2f}s")
