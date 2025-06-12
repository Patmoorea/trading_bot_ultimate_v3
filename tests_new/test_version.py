"""
Version tests
Version 1.0.0 - Created: 2025-05-19 02:22:15 by Patmoorea
"""

from src.config.versions import get_version, get_version_info

def test_version():
    """Test version string format"""
    version = get_version()
    assert isinstance(version, str)
    assert len(version.split('.')) >= 3

def test_version_info():
    """Test version info structure"""
    info = get_version_info()
    assert isinstance(info, dict)
    assert all(k in info for k in ['major', 'minor', 'patch', 'build', 'release'])
