"""Dashboard interactif avec tests intégrés"""
import plotly.graph_objects as go
from src.core_merged.config import Config

class TradingDashboard:
    def __init__(self):
        self.fig = go.Figure()
        self.config = Config()

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

    def render(self):
        """Génération du layout"""
        self.fig.update_layout(
            title='Analyse Technique',
            xaxis_title='Date',
            yaxis_title='Prix',
            template=self.config.theme
        )
        return self.fig
