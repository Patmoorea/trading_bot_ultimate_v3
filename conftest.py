import pytest
import sys

@pytest.fixture(autouse=True)
def torch_patch_fixture():
    """Applique le patch Torch automatiquement pour tous les tests"""
    try:
        from src.utils.torch_patch import patch_torch
        patch_torch()
    except ImportError:
        pass
