"""
Version configuration
Version 1.0.0 - Created: 2025-05-19 02:23:30 by Patmoorea
"""

VERSION = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'build': '2025051902',
    'release': 'stable'
}

def get_version():
    """Get version string"""
    v = VERSION
    return f"{v['major']}.{v['minor']}.{v['patch']}-{v['release']}+{v['build']}"

def get_version_info():
    """Get version information"""
    return VERSION.copy()
