import tensorflow as tf

def force_cpu():
    """Forcer l'exécution sur CPU"""
    tf.config.set_visible_devices([], 'GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('Mode CPU forcé')
