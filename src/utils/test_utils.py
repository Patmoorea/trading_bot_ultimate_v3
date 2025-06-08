"""
Test utilities
Version 1.0.0 - Created: 2025-05-19 03:11:27 by Patmoorea
"""

from unittest.mock import MagicMock

class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

class MockBot:
    def __init__(self, *args, **kwargs):
        self.send_message = AsyncMock(return_value=True)
        self.args = args
        self.kwargs = kwargs
