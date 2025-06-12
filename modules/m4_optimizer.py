import os

from dotenv import load_dotenv


class M4Optimizer:
    def __init__(self):
        load_dotenv()
        self.config = {
            "batch_size": int(os.getenv("BATCH_SIZE", 64)),
            "use_metal": os.getenv("USE_METAL") == "true",
            "use_numba": os.getenv("USE_NUMBA") == "true",
        }

    def configure_torch(self):
        if self.config["use_metal"]:
            import torch

            torch.set_default_device("mps")
            torch.backends.mps.optimize_enabled = True
