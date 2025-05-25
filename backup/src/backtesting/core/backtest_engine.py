"""
Moteur de backtesting - Created: 2025-05-17 23:06:45
@author: Patmoorea
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict = {}
        self.trades: List[Dict] = []
        self.metrics: Dict = {}
        
    def run_backtest(self, data: pd.DataFrame, strategy_func, **params) -> Dict:
        self.reset()
        signals = strategy_func(data, **params)
        
        for timestamp, row in data.iterrows():
            signal = signals.loc[timestamp]
            if signal > 0 and not self.positions:
                self._open_position('LONG', row['close'], timestamp)
            elif signal < 0 and self.positions:
                self._close_position(row['close'], timestamp)
                
        return self.calculate_metrics()
        
    def _open_position(self, direction: str, price: float, timestamp: datetime):
        size = self.current_capital * 0.95  # 95% du capital
        self.positions = {
            'direction': direction,
            'entry_price': price,
            'size': size,
            'entry_time': timestamp
        }
        
    def _close_position(self, price: float, timestamp: datetime):
        if not self.positions:
            return
            
        pnl = (price - self.positions['entry_price']) * self.positions['size']
        if self.positions['direction'] == 'SHORT':
            pnl = -pnl
            
        self.trades.append({
            'entry_time': self.positions['entry_time'],
            'exit_time': timestamp,
            'entry_price': self.positions['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'direction': self.positions['direction']
        })
        
        self.current_capital += pnl
        self.positions = {}
        
    def calculate_metrics(self) -> Dict:
        if not self.trades:
            return {}
            
        pnls = [trade['pnl'] for trade in self.trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        self.metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
        }
        
        return self.metrics
        
    def _calculate_max_drawdown(self) -> float:
        capital_curve = [self.initial_capital]
        for trade in self.trades:
            capital_curve.append(capital_curve[-1] + trade['pnl'])
        capital_curve = pd.Series(capital_curve)
        
        rolling_max = capital_curve.expanding().max()
        drawdowns = (capital_curve - rolling_max) / rolling_max
        return drawdowns.min()
        
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if not self.trades:
            return 0
            
        returns = pd.Series([trade['pnl'] for trade in self.trades])
        excess_returns = returns.mean() - risk_free_rate/252
        return np.sqrt(252) * excess_returns / returns.std()
        
    def reset(self):
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.metrics = {}
