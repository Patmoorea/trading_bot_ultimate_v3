
from src.config.versions import get_version

def test_version():
    version = get_version()
    assert isinstance(version, str)
    assert len(version.split('.')) >= 3

