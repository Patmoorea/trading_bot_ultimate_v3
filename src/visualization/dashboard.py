"""Dashboard interactif avec tests intégrés"""
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
import pandas as pd
from datetime import datetime
import websockets
import json
from typing import Dict, List
import queue
import logging
import psutil
from dataclasses import dataclass
from dash import html

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    liquidation_price: float
    unrealized_pnl: float
    timestamp: datetime

class TradingDashboard:
    def __init__(self, update_interval: int = 1000, max_history: int = 10000):
        self.fig = go.Figure()
        self.config = Config()
        self.update_interval = update_interval
        self.max_history = max_history
        self.active_positions: Dict[str, Position] = {}
        self.trades_stream = queue.Queue()
        self.total_pnl = 1000.0
        self.position_risk = {
            'BTC/USDT': {
                'liquidation_distance': 0.0009,
                'unrealized_pnl': 0.0,
                'exposure': 0.0
            }
        }
        self.pnl_history = pd.DataFrame(columns=['timestamp', 'total_pnl'])
        self.logger = logging.getLogger('TradingDashboard')
        self.handler = logging.StreamHandler()
        self.logger.addHandler(self.handler)
        self.connected = False
        self.ws = None

    async def start_data_stream(self, url: str):
        """Démarrage du flux de données"""
        return await self.connect_websocket(url)

    async def connect_websocket(self, url: str):
        try:
            self.ws = await websockets.connect(url)
            self.connected = True
            return True
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            return False

    def update_plot(self, data):
        """Mise à jour sécurisée du graphique"""
        if not data:
            return {"status": "error", "message": "Données manquantes"}
        
        try:
            self.fig.add_trace(go.Candlestick(
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            ))
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def update_trades(self, trade_data: Dict):
        try:
            self.trades_stream.put(trade_data)
            self._update_position_from_trade(trade_data)
            return self.total_pnl
        except Exception as e:
            self.logger.error(f"Error updating trades: {e}")
            return None

    def _update_position_from_trade(self, trade: Dict):
        symbol = trade.get('symbol')
        if symbol not in self.active_positions:
            self.active_positions[symbol] = Position(
                symbol=symbol,
                size=trade.get('size', 0),
                entry_price=trade.get('price', 0),
                current_price=trade.get('price', 0),
                liquidation_price=trade.get('liquidation_price', 0),
                unrealized_pnl=0,
                timestamp=datetime.now()
            )

    async def handle_market_data(self, data: Dict):
        """Méthode publique pour gérer les données de marché"""
        return await self._handle_market_data(data)

    async def _handle_market_data(self, data: Dict):
        try:
            if 'price' in data and 'symbol' in data:
                self.update_risk_metrics(data)
            return True
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
            return False

    def update_risk_metrics(self, market_data: Dict):
        """Méthode publique pour mettre à jour les métriques de risque"""
        symbol = market_data['symbol']
        if symbol in self.position_risk:
            self.position_risk[symbol]['unrealized_pnl'] = (
                market_data['price'] - self.active_positions[symbol].entry_price
            ) * self.active_positions[symbol].size

    def get_memory_usage(self):
        """Retourne l'utilisation mémoire en MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def create_pnl_chart(self):
        if self.pnl_history.empty:
            return go.Figure()
        
        new_df = pd.concat([self.pnl_history, pd.DataFrame({
            'timestamp': [datetime.now()],
            'total_pnl': [self.total_pnl]
        })], ignore_index=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=new_df['timestamp'],
            y=new_df['total_pnl'],
            mode='lines',
            name='PnL'
        ))
        return fig

    def _create_risk_display(self):
        """Création de l'affichage des risques (version privée)"""
        return self.create_risk_display()

    def create_risk_display(self):
        risk_html = html.Div([
            html.H3("Risk Metrics"),
            html.Div([
                html.P(f"Position Risk: {self.position_risk}"),
                html.P(f"Total PnL: {self.total_pnl}")
            ])
        ])
        return risk_html

    def _create_trading_stats(self):
        """Création des statistiques de trading (version privée)"""
        return self.create_trading_stats()

    def create_trading_stats(self):
        stats_html = html.Div([
            html.H3("Trading Statistics"),
            html.Div([
                html.P(f"Active Positions: {len(self.active_positions)}"),
                html.P(f"Memory Usage: {self.get_memory_usage():.2f} MB")
            ])
        ])
        return stats_html

    def render(self):
        """Génération du layout"""
        self.fig.update_layout(
            title='Analyse Technique',
            xaxis_title='Date',
            yaxis_title='Prix',
            template=self.config.theme
        )
        return self.fig
