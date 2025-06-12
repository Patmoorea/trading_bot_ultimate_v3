"""
Configuration for M1/M2/M4 Mac optimization
Last updated: 2025-05-27 16:18:32 UTC by Patmoorea
"""

TENSORFLOW_CONFIG = {
    'optimizer': 'tf.keras.optimizers.legacy.Adam',  # Optimisé pour M4
    'gpu_memory_growth': True,
    'metal_enabled': True
}

TEST_CONFIG = {
    'max_workers': 10,  # Nombre de cœurs M4
    'memory_limit': '12G',  # Sur 16Go total
    'metal_device': 'Apple M4'
}

SYSTEM_INFO = {
    'cpu': 'Apple M4',
    'cores': 10,
    'ram': '16 Go',
    'gpu': 'Apple M4',
    'os': 'macOS 15.3.2',
    'kernel': 'Darwin 24.3.0 arm64'
}
