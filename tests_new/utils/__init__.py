import pytest
import asyncio
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

@dataclass
class TestConfig:
    telegram_token: str = "test_token"
    telegram_chat_id: str = "test_chat_id"
    performance_log_dir: str = "logs/performance/"
    notification_interval: int = 60
    debug: bool = True

TEST_CONFIG = TestConfig()

class BaseTestCase:
    @classmethod
    def get_test_data(cls, *args, **kwargs):
        """Generate test data with specified size and optional parameters"""
        size = kwargs.get('size', 100)
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=size, freq='H'),
            'price': np.random.uniform(30000, 35000, size),
            'volume': np.random.uniform(1, 10, size),
            'close': np.random.uniform(30000, 35000, size),
            'open': np.random.uniform(30000, 35000, size),
            'high': np.random.uniform(31000, 36000, size),
            'low': np.random.uniform(29000, 34000, size)
        })

class AsyncBaseTestCase(BaseTestCase):
    config = TEST_CONFIG
