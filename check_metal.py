import tensorflow as tf
import platform

print("=== Configuration Apple M4 ===")
print(f"macOS {platform.mac_ver()[0]}")
print(f"Processeur: {platform.processor()}")
print(f"Cores: {platform.processor().count('cpu')}")

gpus = tf.config.list_physical_devices('GPU')
print("\n=== Accélération GPU ===")
print(f"GPU détectés: {len(gpus)}")
if gpus:
    print(f"Type GPU: {gpus[0].device_type}")
    print(f"Nom GPU: {gpus[0].name}")
