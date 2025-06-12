import tensorflow as tf

def check_gpu():
    print("Version TensorFlow:", tf.__version__)
    print("GPU disponibles:", tf.config.list_physical_devices('GPU'))
    print("GPU Metal activé:", tf.test.is_built_with_metal_support())

if __name__ == "__main__":
    check_gpu()
