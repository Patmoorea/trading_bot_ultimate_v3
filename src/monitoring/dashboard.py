#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from datetime import datetime

class TradingDashboard:
    def __init__(self):
        self.title = "Trading Bot Dashboard"
        
    def display(self, data):
        st.title(self.title)
        st.write(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.json(data)

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.display({"status": "ready"})
