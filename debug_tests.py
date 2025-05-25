import pytest
import os

os.environ['PYTHONPATH'] = os.getcwd()
pytest.main(['tests/unit/', '-v', '--cov=src.core', '--cov-report=term'])
