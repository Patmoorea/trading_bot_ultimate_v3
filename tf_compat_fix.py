import numpy as np
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
