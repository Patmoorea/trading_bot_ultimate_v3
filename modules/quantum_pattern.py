import pennylane as qml
from pennylane import numpy as np


class QuantumAnalyzer:
    def __init__(self, n_qubits=4):
        self.n_qubits = min(n_qubits, 4)  # Limité pour M4
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev)
        def circuit(inputs):
            for i in range(self.n_qubits):
                qml.RY(np.pi * inputs[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def analyze(self, data):
        """Wrapper sécurisé"""
        try:
            return self.circuit(data[: self.n_qubits])
        except BaseException:
            return np.zeros(self.n_qubits)
