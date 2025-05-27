import pytest
from datetime import datetime

@pytest.fixture(scope="session")
def timestamp():
    return datetime(2025, 5, 27, 16, 24, 21)

@pytest.fixture(scope="session")
def user():
    return "Patmoorea"

@pytest.fixture(scope="session")
def system_info():
    return {
        "cpu": "Apple M4",
        "os": "macOS 15.3.2",
        "python": "3.11.9"
    }
