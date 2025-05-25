import pytest
from tests_new.base_test import BaseTest
import importlib

class TestImports(BaseTest):
    @pytest.mark.parametrize("module", [
        "numpy",
        "pandas",
        "ccxt"
    ])
    def test_required_packages(self, module):
        """Test that critical packages can be imported"""
        try:
            importlib.import_module(module.replace("-", "_"))
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {str(e)}")

    @pytest.mark.skip(reason="Optional packages not required for basic functionality")
    @pytest.mark.parametrize("module", [
        "tensorflow",
        "ta",
        "python-telegram-bot",
        "python-dotenv"
    ])
    def test_optional_packages(self, module):
        """Test that optional packages can be imported"""
        try:
            importlib.import_module(module.replace("-", "_"))
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {str(e)}")

    def test_local_imports(self):
        """Test that local modules can be imported"""
        basic_modules = [
            "config",
            "src.analysis.technical.indicators"
        ]
        for module in basic_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Failed to import {module}: {str(e)}")
