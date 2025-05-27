#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
from datetime import datetime

class TradingDashboard:
    def __init__(self):
        self.title = "Trading Bot Dashboard"
        
    def display(self, data):
        # Affichage original (conservé)
        st.title(self.title)
        st.write(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.json(data)
        
        # Nouvel affichage style terminal
        st.markdown("""
        ```bash
        $ date -u
        {}

        $ whoami
        {}

        $ cat /var/log/trading_bot/status
        ----------------------------------------
        BOT STATUS: {}
        LAST UPDATE: {}

        $ cat /var/log/trading_bot/signals
        ----------------------------------------
        QSVM SIGNAL: {}
        NEWS SENTIMENT: {}
        MARKET REGIME: {}

        $ cat /var/log/trading_bot/positions
        ----------------------------------------
        CURRENT POSITIONS:
        {}
        ```
        """.format(
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "Patmoorea",
            data.get('status', 'SCANNING...'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            data.get('qsvm_signal', 'N/A'),
            data.get('sentiment', 'N/A'),
            data.get('regime', 'N/A'),
            data.get('positions', 'NO ACTIVE POSITIONS')
        ))

if __name__ == "__main__":
    dashboard = TradingDashboard()
    # Test avec des données d'exemple
    dashboard.display({
        "status": "ACTIVE",
        "qsvm_signal": "BUY",
        "sentiment": "POSITIVE",
        "regime": "TRENDING",
        "positions": "BTC/USDT: LONG @ 35000"
    })
