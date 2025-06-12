import argparse
from modules.quantum_pattern import QuantumAnalyzer
from modules.pair_manager import get_trading_pairs
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum", action="store_true")
    args = parser.parse_args()

    pairs = get_trading_pairs()
    print(f"Backtest démarré à {datetime.now()}")
    print(f"Pairs analysées: {pairs}")

    if args.quantum:
        qa = QuantumAnalyzer(n_qubits=4)
        print("Mode quantum activé | Résultat test:", qa.analyze([0.1, 0.2, 0.3, 0.4]))
    else:
        print("Mode classique")

if __name__ == "__main__":
    main()

def run_quantum_strategy():
    from strategies.quantum_strat import QuantumStrategy
    qs = QuantumStrategy()
    decision = qs.should_buy({'features': [0.1, 0.2, 0.3, 0.4]})
    print(f"Decision de trading: {'ACHAT' if decision else 'NEUTRE'}")
