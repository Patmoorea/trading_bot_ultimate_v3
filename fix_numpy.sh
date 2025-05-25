#!/bin/bash
echo "Correction des dépendances NumPy..."

# Désinstallation des versions problématiques
pip uninstall -y numpy pandas tensorflow keras

# Installation des versions compatibles
pip install "numpy<2" "pandas<2" "tensorflow-macos<3" "keras<3"

# Vérification
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
