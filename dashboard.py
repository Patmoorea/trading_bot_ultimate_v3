import streamlit as st
import time
from strategies.quantum_strat import QuantumStrategy

st.title("Live Trading Dashboard")
qs = QuantumStrategy()

placeholder = st.empty()
while True:
    decision = qs.should_buy({'features': [0.1,0.2,0.3,0.4]})
    placeholder.metric("Decision", "ACHAT" if decision else "NEUTRE")
    time.sleep(5)
