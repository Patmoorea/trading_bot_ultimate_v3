from TradingBot_Advanced.arbitrage.quantum import QuantumArbitrage
from trading_bot_ultimate.arbitrage_engine import ClassicArbitrage

class HybridArbitrage(QuantumArbitrage, ClassicArbitrage):
    """Combine les deux approches"""
