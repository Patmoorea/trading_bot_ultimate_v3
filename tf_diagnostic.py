import sys
import pkg_resources
import numpy as np

print("=== Diagnostic TensorFlow ===")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")

try:
    import tensorflow as tf
    print(f"\nTensorFlow: {tf.__version__}")
    print(f"tf.io: {'Disponible' if hasattr(tf, 'io') else 'Manquant'}")
    print(f"Keras: {tf.keras.__version__ if hasattr(tf, 'keras') else 'Manquant'}")
    print("\nGPU détectés:")
    print(tf.config.list_physical_devices('GPU'))
    
    # Test basique
    print("\nTest d'exécution...")
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    print("Modèle créé avec succès")
except Exception as e:
    print(f"\nERREUR: {str(e)}")

print("\nPackages installés:")
for pkg in ['tensorflow-macos', 'tensorflow-metal', 'tensorboard']:
    try:
        print(f"{pkg}: {pkg_resources.get_distribution(pkg).version}")
    except:
        print(f"{pkg}: Non installé")
