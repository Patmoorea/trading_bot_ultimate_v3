import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProvider:
    async def get_data(self, *args, **kwargs):
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'price': np.random.uniform(30000, 35000, 100),
            'volume': np.random.uniform(1, 10, 100)
        })

class TestDataProvider:
    @pytest.fixture
    def data_provider(self):
        return DataProvider()

    @pytest.mark.asyncio
    async def test_data_ranges(self, data_provider):
        data = await data_provider.get_data()
        assert len(data) == 100
        assert all(col in data.columns for col in ['timestamp', 'price', 'volume'])

    @pytest.mark.asyncio
    async def test_data_validation(self, data_provider):
        data = await data_provider.get_data()
        assert not data.empty
        assert data['price'].min() >= 0
        assert data['volume'].min() >= 0

    @pytest.mark.asyncio
    async def test_timestamp_ordering(self, data_provider):
        data = await data_provider.get_data()
        assert (data['timestamp'].diff()[1:] > timedelta(0)).all()
