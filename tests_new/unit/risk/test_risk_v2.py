import pytest
from src.core.risk_management import RiskManager
from tests_new.base_test import BaseTest

class TestRisk(BaseTest):
    def test_initialization(self):
        """Test risk manager initialization"""
        risk_manager = RiskManager()
        assert risk_manager.max_position_size == 0.1
        assert risk_manager.stop_loss_pct == 0.02
        assert risk_manager.max_drawdown == 0.1
        assert risk_manager.risk_per_trade == 0.01
        
    def test_position_size_calculation(self):
        """Test position sizing logic"""
        risk_manager = RiskManager()
        capital = 10000
        price = 100
        
        # Basic calculation without volatility
        base_size = risk_manager.calculate_position_size(capital, price)
        assert base_size <= capital * risk_manager.max_position_size
        
        # Test with volatility
        vol_size = risk_manager.calculate_position_size(capital, price, volatility=0.5)
        assert vol_size < base_size  # Volatility should reduce position size
        
        # Test size reduction with increasing volatility
        prev_size = float('inf')
        for vol in [0.1, 0.3, 0.5, 0.7, 0.9]:
            curr_size = risk_manager.calculate_position_size(capital, price, volatility=vol)
            assert curr_size < prev_size, f"Size should decrease with volatility {vol}"
            prev_size = curr_size
        
    def test_risk_limits(self):
        """Test risk limit monitoring"""
        risk_manager = RiskManager()
        capital = 10000
        
        # Test within limits
        ok, msg = risk_manager.check_risk_limits(-500, capital)
        assert ok is True
        
        # Test exceeding drawdown
        ok, msg = risk_manager.check_risk_limits(-1100, capital)
        assert ok is False
        assert "drawdown" in msg.lower()
        
    def test_position_tracking(self):
        """Test position tracking functionality"""
        risk_manager = RiskManager()
        
        # Add position
        risk_manager.add_position("BTC/USD", 0.05)
        assert "BTC/USD" in risk_manager.positions
        assert risk_manager.positions["BTC/USD"] == 0.05
        
        # Add to existing position
        risk_manager.add_position("BTC/USD", 0.03)
        assert risk_manager.positions["BTC/USD"] == 0.08
        
        # Remove position
        risk_manager.remove_position("BTC/USD")
        assert "BTC/USD" not in risk_manager.positions
        
    def test_stop_loss_calculation(self):
        """Test stop loss price calculation"""
        risk_manager = RiskManager()
        entry_price = 100
        
        # Long position
        sl_long = risk_manager.get_stop_loss_price(entry_price, True)
        assert sl_long == pytest.approx(entry_price * 0.98)  # 2% below entry
        
        # Short position
        sl_short = risk_manager.get_stop_loss_price(entry_price, False)
        assert sl_short == pytest.approx(entry_price * 1.02)  # 2% above entry

    def test_drawdown_management(self):
        """Test drawdown tracking and reset"""
        risk_manager = RiskManager()
        capital = 10000
        
        # Test drawdown tracking
        risk_manager.check_risk_limits(-500, capital)  # 5% drawdown
        assert risk_manager.current_drawdown == 0.05
        
        risk_manager.check_risk_limits(-700, capital)  # 7% drawdown
        assert risk_manager.current_drawdown == 0.07
        
        risk_manager.check_risk_limits(-300, capital)  # Smaller loss doesn't reduce drawdown
        assert risk_manager.current_drawdown == 0.07
        
        # Test reset
        risk_manager.reset_drawdown()
        assert risk_manager.current_drawdown == 0.0
