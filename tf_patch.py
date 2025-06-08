import tensorflow as tf

# Correctifs critiques
if not hasattr(tf, '__spec__'):
    tf.__spec__ = "tensorflow"

if not hasattr(tf, '__internal__'):
    class MockInternal:
        tracking = type('', (), {'no_automatic_dependency_tracking': lambda f: f})()
    tf.__internal__ = MockInternal()

if not hasattr(tf.keras, '__version__'):
    tf.keras.__version__ = tf.__version__

print("TensorFlow configuré avec succès")
