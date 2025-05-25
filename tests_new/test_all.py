"""
General tests
Version 1.0.0 - Created: 2025-05-19 02:38:15 by Patmoorea
"""

import pytest
from src.config.constants import CURRENT_USER
from src.config.versions import get_version

@pytest.fixture
def current_user():
    return CURRENT_USER

def test_version(current_user):
    version = get_version()
    assert isinstance(version, str)
    assert len(version.split('.')) >= 3
    assert current_user == "Patmoorea"

class TestBasic:
    def test_simple(self):
        assert True
