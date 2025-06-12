#!/bin/bash
# Installation des dépendances
pip install --upgrade gymnasium stable_baselines3 tensorflow-metal

# Correction des fichiers de test
cat <<EOL > tests/ai/test_hybrid.py
[paste le contenu corrigé ci-dessus]
EOL

# Vérification finale
python -m pytest tests/ai/test_hybrid.py -v
