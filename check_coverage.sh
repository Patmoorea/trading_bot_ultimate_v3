#!/bin/bash
echo "=== LANCEMENT DES TESTS AVEC COUVERTURE ==="

# Exécution des tests avec couverture
pytest tests/unit/test_technical_analysis.py -v \
  --cov=src/analysis/technical/advanced \
  --cov-report=term-missing \
  --cov-fail-under=80

# Génération du rapport HTML
pytest --cov=src --cov-report=html

echo "=== RAPPORT DISPONIBLE DANS htmlcov/index.html ==="
