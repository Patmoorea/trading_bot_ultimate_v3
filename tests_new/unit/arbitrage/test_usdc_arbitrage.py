import pytest
from tests_new.base_test import BaseTest
from modules.arbitrage_engine import ArbitrageEngine
from decimal import Decimal
from tests_new.utils import AsyncMockExchange
import asyncio

class TestUSDCArbitrage(BaseTest):
    """Tests pour les nouvelles fonctionnalités d'arbitrage USDC"""
    
    def setUp(self):
        super().init_test()
        self.engine = ArbitrageEngine()
        
        # Remplacer la méthode check_opportunities_v2 pour les tests
        async def mock_check_opportunities():
            return [
                {"pair": "BTC/USDC", "spread": 0.008, "exchanges": ["binance", "kraken"]},
                {"pair": "ETH/USDC", "spread": 0.005, "exchanges": ["binance", "coinbase"]},
                {"pair": "BTC/USDT", "spread": 0.007, "exchanges": ["binance", "bybit"]},
                {"pair": "SOL/USDC", "spread": 0.011, "exchanges": ["kraken", "huobi"]}
            ]
        self.engine.check_opportunities_v2 = mock_check_opportunities
        
        # Remplacer la méthode fetch_order_book pour les tests
        async def mock_fetch_order_book(pair):
            if "BTC" in pair:
                return {
                    "bids": [(30000, 5.2), (29990, 8.7)],
                    "asks": [(30010, 4.3), (30020, 10.1)]
                }
            else:
                return {
                    "bids": [(2000, 15.0), (1990, 22.5)],
                    "asks": [(2010, 8.5), (2020, 30.0)]
                }
        self.engine.fetch_order_book = mock_fetch_order_book
    
    @pytest.mark.asyncio
    async def test_find_usdc_opportunities(self):
        """Test que la méthode renvoie uniquement les paires USDC"""
        opportunities = await self.engine.find_usdc_arbitrage()
        
        assert len(opportunities) == 3
        for opp in opportunities:
            assert "USDC" in opp["pair"]
        
        # Vérifie qu'elles sont triées par spread décroissant
        assert opportunities[0]["pair"] == "SOL/USDC"
        assert opportunities[0]["spread"] == 0.011
        
    @pytest.mark.asyncio
    async def test_estimate_slippage(self):
        """Test l'estimation du slippage"""
        opportunity = {
            "pair": "BTC/USDC",
            "spread": 0.008,
            "exchanges": ["binance", "kraken"]
        }
        
        result = await self.engine.estimate_slippage(opportunity)
        
        assert "estimated_slippage" in result
        assert "net_spread" in result
        assert result["net_spread"] == result["spread"] - result["estimated_slippage"]
        assert result["estimated_slippage"] == 0.0005  # Car volume > 50
