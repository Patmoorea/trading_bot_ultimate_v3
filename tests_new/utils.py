"""
Test utilities
Version 1.0.0 - Created: 2025-05-19 05:13:33 by Patmoorea
"""

from decimal import Decimal
from typing import Dict, Any
import pandas as pd

# Test timestamp that will be used consistently across all tests
TEST_TIMESTAMP = pd.Timestamp('2025-05-19 05:13:33+00:00')

class AsyncMockExchange:
    """Mock exchange class for testing"""
    def __init__(self, name="TestExchange", price_bias=0):
        self.name = name.lower()
        self.price_bias = Decimal(str(price_bias))

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Mock get_ticker method"""
        base_price = Decimal('30000') + self.price_bias
        return {
            'bid': base_price - Decimal('50'),
            'ask': base_price + Decimal('50'),
            'last': base_price,
            'volume': Decimal('1.5'),
            'timestamp': TEST_TIMESTAMP.isoformat()
        }
