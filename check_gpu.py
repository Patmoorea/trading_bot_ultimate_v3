import tensorflow as tf
print("GPU disponibles:", tf.config.list_physical_devices('GPU'))
print("Built with Metal:", tf.test.is_built_with_metal_support())
