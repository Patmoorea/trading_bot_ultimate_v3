from config.quantum_params import BUY_THRESHOLD, N_QUBITS
from modules.quantum_pattern import QuantumAnalyzer
from modules.quantum_fallback import QuantumAnalyzer as FallbackAnalyzer

class QuantumStrategy:
    def __init__(self):
        try:
            self.analyzer = QuantumAnalyzer(n_qubits=N_QUBITS)
            self.quantum_enabled = True
        except:
            self.analyzer = FallbackAnalyzer(n_qubits=N_QUBITS)
            self.quantum_enabled = False

    def should_buy(self, data):
        results = self.analyzer.analyze(data.get('features', [0]*N_QUBITS))
        return results[0] > BUY_THRESHOLD
