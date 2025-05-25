
import torch
import tensorflow as tf

def configure_m4():
    # Configuration TensorFlow Metal
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Configuration PyTorch MPS
    torch.backends.mps.is_available = True
    torch.backends.mps.is_built = True
    torch.set_default_device('mps')
