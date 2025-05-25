import streamlit as st
import plotly.graph_objects as go

class TradingDashboard:
    def __init__(self, data):
        self.data = data

    def render(self):
        st.title("Live Trading Dashboard")
        fig = go.Figure(
            data=[go.Candlestick(
                x=self.data['time'],
                open=self.data['open'],
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close']
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
