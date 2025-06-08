#!/bin/bash
# Nettoyage complet
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +

# Installation
pip install numpy pytest pytest-cov >/dev/null 2>&1

# Tests
echo "=== TESTS AUTOMATISÉS ==="
pytest tests/core/test_technical.py -v --cov=src.analysis.technical

# Vérification
echo "\n=== VÉRIFICATION MANUELLE ==="
python -c "
import numpy as np
from src.analysis.technical import TechnicalAnalyzer

ta = TechnicalAnalyzer()
print('1. Hausse pure (6 périodes):', ta.calculate_rsi([100, 101, 102, 103, 104, 105]))
print('2. Baisse pure (6 périodes):', ta.calculate_rsi([105, 104, 103, 102, 101, 100]))
print('3. Stable (20 périodes):', ta.calculate_rsi([100]*20))
print('4. Gain unique:', ta.calculate_rsi([100, 101]))
print('5. Perte unique:', ta.calculate_rsi([101, 100]))
print('6. Deux périodes:', ta.calculate_rsi([100, 101, 100]))
"
