import tensorflow as tf
import sys

# Correctif complet pour les attributs manquants
if not hasattr(tf, '__spec__'):
    tf.__spec__ = "tensorflow"

# Création d'un module __internal__ complet
if not hasattr(tf, '__internal__'):
    class InternalModule:
        class tracking:
            @staticmethod
            def no_automatic_dependency_tracking(func):
                return func
            
        class distribute:
            class strategy_context:
                @staticmethod
                def get_strategy():
                    return None
    
    tf.__internal__ = InternalModule()

# Patch pour les imports compat.v2
sys.modules['tensorflow.compat.v2'] = tf
sys.modules['tensorflow.compat.v2.__internal__'] = tf.__internal__

# Configuration Keras
if not hasattr(tf.keras, '__version__'):
    tf.keras.__version__ = tf.__version__

print("Critical patches applied successfully:")
print(f"- tf.__spec__: {hasattr(tf, '__spec__')}")
print(f"- tf.__internal__: {hasattr(tf, '__internal__')}")
print(f"- tf.keras.__version__: {hasattr(tf.keras, '__version__')}")
