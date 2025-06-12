#!/bin/bash

echo "=== Vérification des imports ==="
python3 -c "
from src.strategies.arbitrage.core import UnifiedArbitrage
from src.strategies.arbitrage.compat import USDCArbitrage, ArbitrageEngine
print('Tous les imports fonctionnent')"

echo "\n=== Test des anciennes interfaces ==="
python3 -c "
from src.strategies.arbitrage.compat import USDCArbitrage
arb = USDCArbitrage({'exchanges': ['binance']})
print('USDCArbitrage:', 'OK' if hasattr(arb, 'scan_all_pairs') else 'Échec')"

echo "\n=== Vérification complète ==="
