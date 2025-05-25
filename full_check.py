import tensorflow as tf
from src.core_merged.gpu_config import configure_gpu
from test_ai_final import main as test_ai
from test_gpu_perf import test_gpu

def full_test():
    print("=== TEST COMPLET DU SYSTÈME ===")
    
    # Test et configuration GPU
    gpu_ok = test_gpu()
    configure_gpu()  # Applique la configuration optimale
    
    # Test IA
    if gpu_ok:
        print("\n=== LANCEMENT TEST IA ===")
        test_ai()
    else:
        print("\n=== TEST SUR CPU ===")
        tf.config.set_visible_devices([], 'GPU')
        test_ai()

if __name__ == "__main__":
    full_test()
