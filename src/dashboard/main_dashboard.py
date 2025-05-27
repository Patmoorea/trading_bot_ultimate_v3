from dataclasses import dataclass
from datetime import datetime
import logging
import streamlit as st

@dataclass
class TradingMetrics:
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades_count: int = 0
    active_positions: int = 0

class EnhancedTradingDashboard:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = TradingMetrics()
        
    def update_metrics(self, metrics_dict):
        self.metrics = TradingMetrics(
            daily_pnl=metrics_dict['daily_pnl'],
            total_pnl=metrics_dict['total_pnl'],
            win_rate=metrics_dict['win_rate'],
            drawdown=metrics_dict['drawdown'],
            sharpe_ratio=metrics_dict['sharpe_ratio'],
            trades_count=metrics_dict['trades_count'],
            active_positions=metrics_dict['active_positions']
        )
        return self.metrics

class NotificationManager:
    def __init__(self):
        self.notifications = []
        
    def add_alert(self, message, level="info", expiry=None):
        self.notifications.append({
            'message': message,
            'level': level,
            'timestamp': datetime.utcnow(),
            'expiry': expiry,
            'read': False
        })
        return len(self.notifications)
        
    def get_active_alerts(self):
        now = datetime.utcnow()
        return [
            n for n in self.notifications 
            if not n['expiry'] or n['expiry'] > now
        ]
