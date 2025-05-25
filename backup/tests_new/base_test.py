import pytest
from config import Config
from decimal import Decimal

class BaseTest:
    @pytest.fixture
    def config(self):
        return Config()
        
    @pytest.fixture
    def base_parameters(self):
        return Config.get_risk_params()
