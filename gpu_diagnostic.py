import tensorflow as tf
import sys
from pathlib import Path

def analyze_gpu_usage():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Aucun GPU détecté", file=sys.stderr)
        return False
    
    print(f"Détection GPU: {gpus}")
    
    try:
        # Test d'allocation mémoire
        with tf.device('/GPU:0'):
            test_tensor = tf.random.normal((10000, 10000))
            print(f"Tensor alloué sur: {test_tensor.device}")
            del test_tensor
        return True
    except Exception as e:
        print(f"Erreur GPU: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    print("=== Diagnostic GPU ===")
    gpu_ok = analyze_gpu_usage()
    sys.exit(0 if gpu_ok else 1)
