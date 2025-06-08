#!/bin/bash

# Nettoyage
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Installation des dépendances de test
pip install pytest-cov >/dev/null 2>&1

# Exécution des tests
echo "=== Lancement des tests techniques ==="
pytest tests/unit/technical/ -v --cov=src --cov-report=term-missing

echo "\n=== Rapport de couverture ==="
coverage report -m

# Vérification de la qualité du code
echo "\n=== Vérification PEP8 ==="
pip install pycodestyle >/dev/null 2>&1
pycodestyle --exclude=venv*,migrations --max-line-length=120 src/
