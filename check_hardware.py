import tensorflow as tf
import platform

print("=== Configuration Système ===")
print(f"OS: {platform.mac_ver()[0]}")
print(f"Processeur: {platform.processor()}")
print(f"Architecture: {platform.machine()}")
print("\n=== Configuration TensorFlow ===")
print(f"Version: {tf.__version__}")
print(f"GPU Disponible: {tf.config.list_physical_devices('GPU')}")
print(f"Accélération Metal: {'OUI' if 'metal' in tf.sysconfig.get_build_info() else 'NON'}")
