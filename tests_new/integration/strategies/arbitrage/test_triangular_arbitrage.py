import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

@pytest.mark.asyncio
class TestTriangularArbitrage:
    @pytest.fixture
    def mock_market_data(self):
        return {
            'BTC/USDC': {
                'bid': 30000.0,
                'ask': 30100.0,
                'timestamp': int(datetime(2025, 5, 27, 7, 23, 0).timestamp() * 1000)
            },
            'ETH/BTC': {
                'bid': 0.068,
                'ask': 0.069,
                'timestamp': int(datetime(2025, 5, 27, 7, 23, 0).timestamp() * 1000)
            },
            'ETH/USDC': {
                'bid': 2100.0,
                'ask': 2110.0,
                'timestamp': int(datetime(2025, 5, 27, 7, 23, 0).timestamp() * 1000)
            }
        }

    async def test_triangular_opportunity_detection(self, mock_market_data):
        # Test avec USDC au lieu de USDT
        initial_amount = Decimal('1000')  # USDC
        
        # USDC -> BTC -> ETH -> USDC
        btc_price = Decimal(str(mock_market_data['BTC/USDC']['ask']))
        btc_amount = (initial_amount / btc_price).quantize(Decimal('0.00000001'))
        
        eth_btc_price = Decimal(str(mock_market_data['ETH/BTC']['ask']))
        eth_amount = (btc_amount / eth_btc_price).quantize(Decimal('0.00000001'))
        
        eth_usdc_price = Decimal(str(mock_market_data['ETH/USDC']['bid']))
        
        # Calculer d'abord le résultat sans quantize
        final_amount_raw = eth_amount * eth_usdc_price
        # Puis quantize avec une précision appropriée pour USDC (2 décimales)
        final_amount = final_amount_raw.quantize(Decimal('0.01'))

        profit_ratio = final_amount / initial_amount
        assert profit_ratio > Decimal('1.001'), f"L'opportunité d'arbitrage devrait être profitable (ratio: {profit_ratio})"
