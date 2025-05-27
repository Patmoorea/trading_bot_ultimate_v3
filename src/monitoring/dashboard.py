import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Trading Bot Monitor", layout="wide")
        
    def render(self, data: Dict[str, Any]):
        """Rendu du dashboard"""
        # Header
        st.title("Trading Bot Monitor")
        
        # Métriques principales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PnL Total", f"${data['total_pnl']:,.2f}")
        with col2:
            st.metric("Win Rate", f"{data['win_rate']*100:.1f}%")
        with col3:
            st.metric("Positions Actives", data['active_positions'])
            
        # Graphiques
        self.plot_portfolio_value(data['portfolio_history'])
        self.plot_trade_distribution(data['trades'])
        
    def plot_portfolio_value(self, history: list):
        """Graphique valeur du portfolio"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[h['timestamp'] for h in history],
            y=[h['value'] for h in history],
            mode='lines',
            name='Portfolio Value'
        ))
        st.plotly_chart(fig)
