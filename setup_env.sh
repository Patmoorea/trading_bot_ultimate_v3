#!/bin/bash
echo "Mise à jour de pip..."
python -m pip install --upgrade pip

echo "Installation des dépendances de base..."
pip install ccxt pandas numpy pytest asyncio scikit-learn

echo "Installation des dépendances IA..."
pip install tensorflow tensorflow-metal

echo "Vérification finale..."
python check_imports.py
