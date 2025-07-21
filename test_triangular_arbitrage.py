import pytest
import modules.triangular_arbitrage
from modules.triangular_arbitrage import TriangularArbitrage


class DummyMoteur:
    def __init__(self, config):
        self.exchanges = {"demo": "EXCHANGE"}
        self.config = config

    async def fetch_orderbook(self, exchange, pair):
        if pair == "BTC/USDT":
            return {"asks": [[30000, 100]], "bids": [[29990, 100]]}
        if pair == "ETH/BTC":
            return {"asks": [[0.055, 50]], "bids": [[0.054, 50]]}
        if pair == "ETH/USDT":
            return {"asks": [[1650, 80]], "bids": [[1640, 80]]}
        return None


@pytest.fixture
def config():
    return {
        "min_profit": 0.001,
        "fee_threshold": 0.002,
        "min_volume": 10,
        "pairs": ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
    }


@pytest.mark.asyncio
async def test_find_triangular_opportunities(config, monkeypatch):
    monkeypatch.setattr(modules.triangular_arbitrage, "MoteurArbitrage", DummyMoteur)
    arb = TriangularArbitrage(config)
    opportunities = await arb.find_triangular_opportunities()
    assert isinstance(opportunities, list)
    assert len(opportunities) > 0
    for opp in opportunities:
        assert "profit_pct" in opp
        assert opp["profit_pct"] > config["min_profit"]
        assert "path" in opp and isinstance(opp["path"], tuple)
        assert "rates" in opp and len(opp["rates"]) in (2, 3)


@pytest.mark.asyncio
async def test_calculate_path_profit(config, monkeypatch):
    monkeypatch.setattr(modules.triangular_arbitrage, "MoteurArbitrage", DummyMoteur)
    arb = TriangularArbitrage(config)
    pairs = ("BTC", "ETH", "USDT")
    orderbooks = {
        "BTC/USDT": {"demo": {"asks": [[30000, 100]], "bids": [[29990, 100]]}},
        "ETH/BTC": {"demo": {"asks": [[0.055, 50]], "bids": [[0.054, 50]]}},
        "ETH/USDT": {"demo": {"asks": [[1650, 80]], "bids": [[1640, 80]]}},
    }
    result = await arb.calculate_path_profit(pairs, "demo", orderbooks)
    assert result is not None
    assert "profit_pct" in result
    assert result["profit_pct"] > 0
