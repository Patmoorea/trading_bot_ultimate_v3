#!/bin/bash

# 1. Corriger les imports de configuration
find . -name "*.py" -exec sed -i '' 's/from src.config import Config/from src.core.config import Config/g' {} +

# 2. Désactiver Numba pour les tests
echo "Config.USE_NUMBA = False" >> src/__init__.py

# 3. Corriger les décorateurs jit problématiques
find src/ -name "*.py" -exec sed -i '' 's/@jit(nopython=Config.USE_NUMBA)//g' {} +

# 4. Lancer les tests corrigés
pytest tests/unit/test_basic.py -v
pytest tests/ai/test_cnn_lstm_updated.py -v
