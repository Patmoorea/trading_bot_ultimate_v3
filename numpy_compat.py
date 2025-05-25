import numpy as np
import sys

# Fix pour les anciennes versions qui attendent np.complex_
if not hasattr(np, 'complex_'):
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        if tf.__version__.startswith('2.'):
            np.complex_ = np.complex128
            sys.modules['tensorflow'].__dict__['np'].complex_ = np.complex128
