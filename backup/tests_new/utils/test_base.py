"""
Base test utilities
Version 1.0.0 - Created: 2025-05-19 05:32:29 by Patmoorea
"""

import pytest
from datetime import datetime
import pytz
from src.utils.datetime_utils import format_timestamp

class BaseTestCase:
    """Base class for all test cases"""
    
    @staticmethod
    def get_test_timestamp():
        """Get the global test timestamp"""
        return pytest.TEST_TIMESTAMP
    
    @staticmethod
    def parse_iso_timestamp(timestamp_str):
        """Parse ISO format timestamp string"""
        return datetime.fromisoformat(timestamp_str)
    
    @staticmethod
    def format_timestamp(dt=None):
        """Format datetime to ISO string"""
        if dt is None:
            dt = pytest.TEST_TIMESTAMP
        return format_timestamp(dt)

class AsyncBaseTestCase(BaseTestCase):
    """Base class for async test cases"""
    
    async def async_setup(self):
        """Setup async test case"""
        pass
    
    async def async_teardown(self):
        """Teardown async test case"""
        pass
