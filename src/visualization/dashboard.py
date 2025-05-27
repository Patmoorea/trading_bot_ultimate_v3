#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List
from queue import Queue
import logging
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class Position:
    """Classe pour représenter une position de trading"""
    symbol: str
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    liquidation_price: Decimal = Decimal('0')
    pnl: Decimal = Decimal('0')
    timestamp: datetime = datetime.utcnow()

class TradingDashboard:
    def __init__(self):
        self.title = "Trading Bot Dashboard"
        self.update_interval = 100  # millisecondes
        self.max_history = 1000    # points de données maximum
        
        # Initialisation des DataFrames avec toutes les colonnes nécessaires
        self.pnl_history = pd.DataFrame(columns=['timestamp', 'total_pnl', 'daily_pnl'])
        self.trades_stream = Queue()
        self.active_positions: Dict[str, Position] = {}
        self.portfolio_risk = {
            'total_exposure': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Configuration du logger
        self.logger = logging.getLogger('TradingDashboard')
        self.total_pnl = 0.0

    def update_trades(self, trade: Dict):
        """Mise à jour des trades"""
        try:
            # Mise à jour du PnL total
            self.total_pnl += trade.get('pnl', 0.0)
            
            # Ajout à l'historique avec toutes les colonnes
            new_row = pd.DataFrame([{
                'timestamp': trade.get('timestamp', datetime.utcnow()),
                'total_pnl': self.total_pnl,
                'daily_pnl': trade.get('pnl', 0.0)
            }])
            
            self.pnl_history = pd.concat([self.pnl_history, new_row], ignore_index=True)
            
            # Garder seulement les derniers points
            if len(self.pnl_history) > self.max_history:
                self.pnl_history = self.pnl_history.tail(self.max_history)
                
        except Exception as e:
            self.logger.error(f"Error updating trades: {str(e)}")

    def update_risk_metrics(self, positions: List[Dict]):
        """Mise à jour des métriques de risque"""
        try:
            total_exposure = Decimal('0')
            max_drawdown = Decimal('0')
            
            for position_data in positions:
                position = Position(
                    symbol=position_data['symbol'],
                    size=Decimal(str(position_data.get('size', '0'))),
                    entry_price=Decimal(str(position_data.get('entry_price', '0'))),
                    current_price=Decimal(str(position_data.get('current_price', '0'))),
                    liquidation_price=Decimal(str(position_data.get('liquidation_price', '0')))
                )
                
                # Calcul de l'exposition
                position_value = position.size * position.current_price
                total_exposure += position_value
                
                # Calcul du drawdown pour cette position
                if position.entry_price > 0:
                    drawdown = (position.entry_price - position.current_price) / position.entry_price * 100
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Calcul du PnL
                position.pnl = (position.current_price - position.entry_price) * position.size
                
                # Mise à jour des positions actives
                self.active_positions[position.symbol] = position
            
            # Mise à jour du portfolio risk
            self.portfolio_risk.update({
                'total_exposure': float(total_exposure),
                'max_drawdown': float(max_drawdown),
                # Note: Sharpe ratio nécessite plus de données historiques
            })
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {str(e)}")

    async def _handle_market_data(self, data: Dict):
        """Gestion des données de marché"""
        try:
            if data.get('type') == 'trade':
                self.trades_stream.put(data)
        except Exception as e:
            self.logger.error(f"Error handling market data: {str(data.get('symbol', 'Unknown'))}")

    def display(self):
        """Affichage du dashboard"""
        st.title(self.title)
        
        # Affichage des métriques principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total PnL", f"${self.total_pnl:,.2f}")
        with col2:
            st.metric("Exposure", f"${self.portfolio_risk['total_exposure']:,.2f}")
        with col3:
            st.metric("Max Drawdown", f"{self.portfolio_risk['max_drawdown']:.2f}%")
        
        # Graphique PnL
        if not self.pnl_history.empty:
            st.line_chart(self.pnl_history.set_index('timestamp')['total_pnl'])
        
        # Positions actives
        st.subheader("Active Positions")
        if self.active_positions:
            positions_df = pd.DataFrame([
                {
                    'Size': pos.size,
                    'Entry Price': pos.entry_price,
                    'Current Price': pos.current_price,
                    'PnL': pos.pnl,
                    'Liquidation': pos.liquidation_price
                } for pos in self.active_positions.values()
            ], index=self.active_positions.keys())
            st.dataframe(positions_df)
        else:
            st.write("No active positions")

        # Dernière mise à jour
        st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.display()

# Evolution 2025-05-26 04:21:12 by Patmoorea
# Ajout des nouvelles fonctionnalités sans suppression des précédentes

@dataclass
class Position:
    """Position dataclass avec nouvelles métriques"""
    symbol: str
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    liquidation_price: Decimal = Decimal('0')
    pnl: Decimal = Decimal('0')  
    timestamp: datetime = datetime.utcnow()
    leverage: Decimal = Decimal('1')
    margin_type: str = 'cross'
    
class TradingDashboard:
    def __init__(self, update_interval: int = 100, max_history: int = 1000):
        # Evolution des paramètres existants
        self.update_interval = update_interval 
        self.max_history = max_history
        self.title = "Trading Bot Dashboard"
        
        # Nouveaux paramètres
        self.websocket_enabled = True
        self.realtime_updates = True
        self._worker_task = None
        self._running = False
        
        # Métriques avancées
        self.portfolio_metrics = {
            'total_exposure': Decimal('0'),
            'max_drawdown': Decimal('0'),
            'sharpe_ratio': Decimal('0'),
            'sortino_ratio': Decimal('0'),
            'win_rate': Decimal('0')
        }
        
    async def start_data_stream(self):
        """Nouveau stream de données temps réel"""
        self._running = True
        while self._running:
            try:
                await asyncio.sleep(self.update_interval / 1000)
            except Exception as e:
                self.logger.error(f"Stream error: {str(e)}")
                
    async def stop_data_stream(self):
        """Arrêt propre du stream"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def update_metrics(self, new_position: Position):
        """Mise à jour des métriques avec la nouvelle position"""
        if new_position.symbol not in self.active_positions:
            self.active_positions[new_position.symbol] = new_position
        else:
            current = self.active_positions[new_position.symbol]
            current.current_price = new_position.current_price
            current.pnl = new_position.pnl
            current.timestamp = datetime.utcnow()

# Evolution 2025-05-26 04:28:46 by Patmoorea
    def _update_metrics(self):
        """Mise à jour complète des métriques de portfolio"""
        if not self.active_positions:
            return
            
        # Exposition totale
        self.portfolio_metrics['total_exposure'] = sum(
            p.size * p.current_price for p in self.active_positions.values()
        )
        
        # Drawdown maximal
        pnl_history = pd.Series([p.pnl for p in self.active_positions.values() if p.pnl])
        if not pnl_history.empty:
            cummax = pnl_history.cummax()
            drawdown = (cummax - pnl_history) / cummax
            self.portfolio_metrics['max_drawdown'] = Decimal(str(drawdown.max()))
            
        # Ratio de Sharpe
        if len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            if len(returns) > 0:
                avg_return = returns.mean()
                std_return = returns.std()
                if std_return > 0:
                    sharpe = (avg_return - 0.02) / std_return * (252 ** 0.5)  # Annualisé
                    self.portfolio_metrics['sharpe_ratio'] = Decimal(str(sharpe))
                    
        # Ratio de Sortino
        if len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0:
                avg_return = returns.mean()
                neg_std = neg_returns.std()
                if neg_std > 0:
                    sortino = (avg_return - 0.02) / neg_std * (252 ** 0.5)
                    self.portfolio_metrics['sortino_ratio'] = Decimal(str(sortino))
                    
        # Win Rate
        if len(pnl_history) > 0:
            wins = sum(1 for pnl in pnl_history if pnl > 0)
            self.portfolio_metrics['win_rate'] = Decimal(str(wins / len(pnl_history)))
            
        # Volatilité
        if len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            vol = returns.std() * (252 ** 0.5)  # Annualisée
            self.portfolio_metrics['volatility'] = Decimal(str(vol))
            
        # Beta (par rapport au marché)
        if 'market_returns' in self.__dict__ and len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            if len(returns) == len(self.market_returns):
                covariance = returns.cov(self.market_returns)
                market_variance = self.market_returns.var()
                if market_variance > 0:
                    beta = covariance / market_variance
                    self.portfolio_metrics['beta'] = Decimal(str(beta))
                    
        # Ratio d'information
        if 'benchmark_returns' in self.__dict__ and len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            if len(returns) == len(self.benchmark_returns):
                excess_returns = returns - self.benchmark_returns
                tracking_error = excess_returns.std()
                if tracking_error > 0:
                    info_ratio = excess_returns.mean() / tracking_error
                    self.portfolio_metrics['information_ratio'] = Decimal(str(info_ratio))
                    
        # Value at Risk (VaR)
        if len(pnl_history) > 0:
            var_95 = np.percentile(pnl_history, 5)
            self.portfolio_metrics['var_95'] = Decimal(str(var_95))
            
        # Expected Shortfall (CVaR)
        if len(pnl_history) > 0:
            var_95 = np.percentile(pnl_history, 5)
            losses = pnl_history[pnl_history <= var_95]
            if len(losses) > 0:
                cvar = losses.mean()
                self.portfolio_metrics['cvar_95'] = Decimal(str(cvar))
                
        # Profit Factor
        if len(pnl_history) > 0:
            gains = sum(pnl for pnl in pnl_history if pnl > 0)
            losses = abs(sum(pnl for pnl in pnl_history if pnl < 0))
            if losses > 0:
                profit_factor = gains / losses
                self.portfolio_metrics['profit_factor'] = Decimal(str(profit_factor))

        # Montant en risque
        risk_amount = sum(
            (p.entry_price - p.liquidation_price).copy_abs() * p.size
            for p in self.active_positions.values()
            if p.liquidation_price is not None
        )
        self.portfolio_metrics['risk_amount'] = Decimal(str(risk_amount))

# Evolution 2025-05-26 04:28:46 by Patmoorea
    def _update_metrics(self):
        """Mise à jour complète des métriques de portfolio"""
        if not self.active_positions:
            return
            
        # Exposition totale
        self.portfolio_metrics['total_exposure'] = sum(
            p.size * p.current_price for p in self.active_positions.values()
        )
        
        # Drawdown maximal
        pnl_history = pd.Series([p.pnl for p in self.active_positions.values() if p.pnl])
        if not pnl_history.empty:
            cummax = pnl_history.cummax()
            drawdown = (cummax - pnl_history) / cummax
            self.portfolio_metrics['max_drawdown'] = Decimal(str(drawdown.max()))
            
        # Ratio de Sharpe
        if len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            if len(returns) > 0:
                avg_return = returns.mean()
                std_return = returns.std()
                if std_return > 0:
                    sharpe = (avg_return - 0.02) / std_return * (252 ** 0.5)  # Annualisé
                    self.portfolio_metrics['sharpe_ratio'] = Decimal(str(sharpe))
                    
        # Ratio de Sortino
        if len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0:
                avg_return = returns.mean()
                neg_std = neg_returns.std()
                if neg_std > 0:
                    sortino = (avg_return - 0.02) / neg_std * (252 ** 0.5)
                    self.portfolio_metrics['sortino_ratio'] = Decimal(str(sortino))
                    
        # Win Rate
        if len(pnl_history) > 0:
            wins = sum(1 for pnl in pnl_history if pnl > 0)
            self.portfolio_metrics['win_rate'] = Decimal(str(wins / len(pnl_history)))
            
        # Volatilité
        if len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            vol = returns.std() * (252 ** 0.5)  # Annualisée
            self.portfolio_metrics['volatility'] = Decimal(str(vol))
            
        # Beta (par rapport au marché)
        if 'market_returns' in self.__dict__ and len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            if len(returns) == len(self.market_returns):
                covariance = returns.cov(self.market_returns)
                market_variance = self.market_returns.var()
                if market_variance > 0:
                    beta = covariance / market_variance
                    self.portfolio_metrics['beta'] = Decimal(str(beta))
                    
        # Ratio d'information
        if 'benchmark_returns' in self.__dict__ and len(pnl_history) > 1:
            returns = pnl_history.pct_change().dropna()
            if len(returns) == len(self.benchmark_returns):
                excess_returns = returns - self.benchmark_returns
                tracking_error = excess_returns.std()
                if tracking_error > 0:
                    info_ratio = excess_returns.mean() / tracking_error
                    self.portfolio_metrics['information_ratio'] = Decimal(str(info_ratio))
                    
        # Value at Risk (VaR)
        if len(pnl_history) > 0:
            var_95 = np.percentile(pnl_history, 5)
            self.portfolio_metrics['var_95'] = Decimal(str(var_95))
            
        # Expected Shortfall (CVaR)
        if len(pnl_history) > 0:
            var_95 = np.percentile(pnl_history, 5)
            losses = pnl_history[pnl_history <= var_95]
            if len(losses) > 0:
                cvar = losses.mean()
                self.portfolio_metrics['cvar_95'] = Decimal(str(cvar))
                
        # Profit Factor
        if len(pnl_history) > 0:
            gains = sum(pnl for pnl in pnl_history if pnl > 0)
            losses = abs(sum(pnl for pnl in pnl_history if pnl < 0))
            if losses > 0:
                profit_factor = gains / losses
                self.portfolio_metrics['profit_factor'] = Decimal(str(profit_factor))

        # Montant en risque
        risk_amount = sum(
            (p.entry_price - p.liquidation_price).copy_abs() * p.size
            for p in self.active_positions.values()
            if p.liquidation_price is not None
        )
        self.portfolio_metrics['risk_amount'] = Decimal(str(risk_amount))
