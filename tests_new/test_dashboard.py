# Created: 2025-05-25 18:39:54 UTC
# Author: Patmoorea
# Project: trading_bot_ultimate

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json

from src.visualization.dashboard import TradingDashboard, Position

@pytest.fixture
def dashboard():
    """Create a dashboard instance for testing"""
    return TradingDashboard(update_interval=100, max_history=1000)

def test_dashboard_initialization(dashboard):
    """Test dashboard initialization"""
    assert dashboard.update_interval == 100
    assert dashboard.max_history == 1000
    assert isinstance(dashboard.active_positions, dict)
    assert isinstance(dashboard.pnl_history, pd.DataFrame)
    assert len(dashboard.pnl_history.columns) == 3
    assert dashboard.running == True

def test_position_dataclass():
    """Test Position dataclass"""
    now = datetime.utcnow()
    pos = Position(
        symbol="BTC/USDT",
        size=1.0,
        entry_price=50000.0,
        current_price=51000.0,
        liquidation_price=45000.0,
        unrealized_pnl=1000.0,
        timestamp=now
    )
    
    assert pos.symbol == "BTC/USDT"
    assert pos.size == 1.0
    assert pos.entry_price == 50000.0
    assert pos.current_price == 51000.0
    assert pos.liquidation_price == 45000.0
    assert pos.unrealized_pnl == 1000.0
    assert pos.timestamp == now

def test_update_trades(dashboard):
    """Test trade updates"""
    trade = {
        "symbol": "BTC/USDT",
        "side": "sell",
        "price": 51000.0,
        "amount": 1.0,
        "entry_price": 50000.0,
        "pnl": 1000.0,
        "timestamp": datetime.utcnow()
    }
    
    dashboard.update_trades(trade)
    assert dashboard.total_pnl == 1000.0
    assert len(dashboard.pnl_history) == 1

def test_risk_metrics_update(dashboard):
    """Test risk metrics updates"""
    positions = [{
        "symbol": "BTC/USDT",
        "size": 1.0,
        "entry_price": 50000.0,
        "current_price": 51000.0,
        "liquidation_price": 45000.0
    }]
    
    dashboard.update_risk_metrics(positions)
    
    risk = dashboard.position_risk["BTC/USDT"]
    assert "liquidation_distance" in risk
    assert "unrealized_pnl" in risk
    assert risk["unrealized_pnl"] == 1000.0
    assert abs(risk["liquidation_distance"] - 0.12) < 0.001

@pytest.mark.asyncio
async def test_market_data_handling(dashboard):
    """Test market data handling"""
    test_data = {
        "type": "trade",
        "symbol": "BTC/USDT",
        "price": 51000.0,
        "amount": 1.0,
        "timestamp": datetime.utcnow().timestamp()
    }
    
    await dashboard._handle_market_data(test_data)
    assert len(list(dashboard.trades_stream.queue)) == 1

def test_memory_usage(dashboard):
    """Test memory usage calculation"""
    memory = dashboard.get_memory_usage()
    assert isinstance(memory, float)
    assert memory > 0

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test websocket connection handling"""
    dashboard = TradingDashboard()
    
    # Mock websocket
    mock_ws = AsyncMock()
    mock_ws.__aenter__.return_value.recv.return_value = json.dumps({
        "type": "trade",
        "symbol": "BTC/USDT",
        "price": 50000,
        "amount": 1
    })
    
    with patch('websockets.connect', return_value=mock_ws):
        task = asyncio.create_task(dashboard.start_data_stream())
        await asyncio.sleep(0.1)  # Allow some time for processing
        dashboard.running = False
        await task
        
        assert mock_ws.__aenter__.called
        assert mock_ws.__aenter__.return_value.recv.called

def test_pnl_chart_creation(dashboard):
    """Test PnL chart creation"""
    # Add some test data
    for i in range(10):
        dashboard.pnl_history = dashboard.pnl_history.append({
            'timestamp': datetime.utcnow() + timedelta(hours=i),
            'total_pnl': i * 100,
            'daily_pnl': 100
        }, ignore_index=True)
    
    fig = dashboard._create_pnl_chart()
    assert len(fig.data) == 2  # Line and bar traces
    assert fig.data[0].name == 'Cumulative P&L'
    assert fig.data[1].name == 'Daily P&L'

def test_risk_display_creation(dashboard):
    """Test risk display creation"""
    dashboard.portfolio_risk = {
        'total_exposure': 100000,
        'max_drawdown': 5.5,
        'sharpe_ratio': 2.1
    }
    
    risk_display = dashboard._create_risk_display()
    assert isinstance(risk_display, html.Div)
    assert len(risk_display.children) > 0

@pytest.mark.asyncio
async def test_error_handling(dashboard):
    """Test error handling in market data processing"""
    bad_data = {"type": "invalid"}
    
    # Should not raise exception
    await dashboard._handle_market_data(bad_data)
    
    # Check error was logged
    assert any("Error handling market data" in record.message 
              for record in dashboard.logger.handler.records)

def test_trading_stats_creation(dashboard):
    """Test trading statistics creation"""
    # Add some test trades
    trades = [
        {"pnl": 100, "timestamp": datetime.utcnow()},
        {"pnl": -50, "timestamp": datetime.utcnow()},
        {"pnl": 75, "timestamp": datetime.utcnow()}
    ]
    
    for trade in trades:
        dashboard.trades_stream.put(trade)
    
    stats = dashboard._create_trading_stats()
    assert isinstance(stats, html.Div)
    assert len(stats.children) == 5  # Total trades, win rate, avg/best/worst trade

if __name__ == '__main__':
    pytest.main([__file__])
