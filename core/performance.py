import tensorflow as tf
import time
import os
import numpy as np

_OPTIMIZED = False
_WARMUP_DONE = False

def optimize_for_m4():
    """Configuration unique pour M4"""
    global _OPTIMIZED
    if _OPTIMIZED:
        return
    
    os.environ['TF_METAL_DISABLE_XLA'] = '1'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.threading.set_intra_op_parallelism_threads(10)
            tf.config.threading.set_inter_op_parallelism_threads(4)
            print('⚡ Optimisation M4 (Metal) - Mémoire dynamique')
            _OPTIMIZED = True
        except RuntimeError as e:
            print('⚠️ Erreur configuration GPU:', e)

def _warmup(matrix_size=1000):
    """Warmup séparé plus intelligent"""
    global _WARMUP_DONE
    if _WARMUP_DONE:
        return
        
    a = tf.random.normal([matrix_size, matrix_size])
    b = tf.random.normal([matrix_size, matrix_size])
    for _ in range(3):
        tf.matmul(a, b).numpy()
    _WARMUP_DONE = True

def benchmark_m4(matrix_size=15000):
    """Benchmark fiable pour M4"""
    optimize_for_m4()
    _warmup(matrix_size)
    
    @tf.function
    def matmul_fn(a, b):
        return tf.matmul(a, b)
    
    a = tf.random.normal([matrix_size, matrix_size])
    b = tf.random.normal([matrix_size, matrix_size])
    
    # Synchronisation alternative
    start = time.perf_counter()
    result = matmul_fn(a, b).numpy()
    duration = time.perf_counter() - start
    
    # Vérification basique
    checksum = np.sum(result)
    print(f'Matrice {matrix_size}x{matrix_size}')
    print(f'Temps: {duration:.4f}s | Checksum: {checksum:.2f}')
    return duration

def advanced_benchmark(matrix_size=20000, runs=5):
    """Benchmark avancé avec statistiques"""
    optimize_for_m4()
    _warmup(matrix_size)
    
    times = []
    for _ in range(runs):
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])
        
        start = time.perf_counter()
        tf.matmul(a, b).numpy()
        times.append(time.perf_counter() - start)
        
    avg = np.mean(times)
    std = np.std(times)
    print(f'{runs} runs sur {matrix_size}x{matrix_size}')
    print(f'Moyenne: {avg:.4f}s ± {std:.4f}s')
    return avg

def monitor_thermal():
    """Monitoring thermique (macOS seulement)"""
    try:
        import subprocess
        result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1'],
                               capture_output=True, text=True)
        print(result.stdout.split('GPU die temperature')[1].split('\n')[0])
    except:
        print('Monitoring thermique non disponible')

def pro_benchmark(matrix_size=20000, runs=5):
    """Benchmark complet avec monitoring"""
    optimize_for_m4()
    _warmup(matrix_size)
    
    print(f'\n🔍 Benchmark professionnel {matrix_size}x{matrix_size}')
    monitor_thermal()
    
    times = []
    for i in range(runs):
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])
        
        start = time.perf_counter()
        c = tf.matmul(a, b).numpy()
        duration = time.perf_counter() - start
        times.append(duration)
        
        checksum = np.sum(c)
        print(f'Run {i+1}: {duration:.4f}s | Checksum: {checksum:.2f}')
        monitor_thermal()
    
    avg = np.mean(times)
    std = np.std(times)
    print(f'\n📊 Résumé: {avg:.4f}s ± {std:.4f}s')
    return avg

def check_performance_threshold(matrix_size=5000, max_time=1.0):
    """Vérifie que les performances sont optimales"""
    temp = get_gpu_temp()
    if temp and temp > 90:  # Seuil température
        print(f'Alerte: Température GPU élevée ({temp}°C)')
        return False
    
    duration = benchmark_m4(matrix_size, verbose=False)
    if duration > max_time:
        print(f'Alerte: Performance GPU insuffisante ({duration:.2f}s)')
        return False
        
    return True
