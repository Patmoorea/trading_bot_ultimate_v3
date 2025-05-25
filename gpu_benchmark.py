import tensorflow as tf
import time

@tf.function
def benchmark():
    matrix_size = 3000
    a = tf.random.normal((matrix_size, matrix_size))
    b = tf.random.normal((matrix_size, matrix_size))
    return tf.matmul(a, b)

print("=== Benchmark GPU ===")
start_time = time.time()
result = benchmark()
print(f"Temps d'exécution: {time.time() - start_time:.2f}s")
print(f"Device utilisé: {result.device}")
