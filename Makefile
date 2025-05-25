quantum-test:
	@echo "\n=== TEST QUANTUM ==="
	python backtest.py --quantum

quantum-decision:
	@echo "\n=== DECISION QUANTUM ==="
	@python3 -c "\
from strategies.quantum_strat import QuantumStrategy;\
qs = QuantumStrategy();\
print(f'Quantum: {qs.quantum_enabled} | Decision: {qs.should_buy({\"features\": [0.1,0.2,0.3,0.4]})}')"
