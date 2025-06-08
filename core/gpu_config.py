import tensorflow as tf
import os

def configure_gpu():
    """Configuration optimale pour Mac M4"""
    # Désactiver XLA problématique
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    tf.config.optimizer.set_jit(False)
    
    # Configurer Metal
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Aucun GPU détecté - Mode CPU activé")
        tf.config.set_visible_devices([], 'GPU')
        return False
    
    try:
        # Mode mémoire dynamique
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    except RuntimeError as e:
        print(f"Erreur configuration GPU: {e}")
        return False
def optimize_performance():
    """Paramètres avancés pour Metal"""
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
    })
