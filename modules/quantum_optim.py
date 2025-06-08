def create_quantum_device():
    try:
        return qml.device("default.qubit", wires=4, backend="lightning.qubit")
    except BaseException:
        return qml.device("default.qubit", wires=4)  # Fallback
