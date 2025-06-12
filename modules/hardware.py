def configure_quantum():
    import os

    import torch

    os.environ.update({"OMP_NUM_THREADS": "8", "PYLON_RUN": "metal"})
    if torch.backends.mps.is_available():
        torch.backends.mps.optimize_operations = True
