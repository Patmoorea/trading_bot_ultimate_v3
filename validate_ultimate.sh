#!/bin/bash
# Nettoyage complet
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +

# Installation des dépendances
pip install pytest pytest-cov >/dev/null 2>&1

# Exécution des tests
echo "=== LANCEMENT DES TESTS ==="
pytest tests/valid/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=80

# Vérification manuelle
echo "=== VERIFICATION FINALE ==="
python -c "
from src.analysis.technical import TechnicalAnalyzer
analyzer = TechnicalAnalyzer()
print('RSI calculé:', 0 <= analyzer.calculate_rsi([1,2,3]*10) <= 100)
print('Cache initialisé:', 'rsi' in analyzer.cache)
"
