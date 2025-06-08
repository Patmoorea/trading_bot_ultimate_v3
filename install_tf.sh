#!/bin/bash
echo "Installation des dépendances manquantes..."

# Désinstallation des versions problématiques
pip uninstall -y tensorflow keras numpy pandas

# Installation des versions stables
pip install --no-cache-dir \
    "numpy<2" \
    "pandas<2" \
    "tensorflow-macos==2.15.0" \
    "keras==2.15.0" \
    "protobuf<4"  # Évite les conflits de version

# Vérification
python -c "
import tensorflow as tf; 
import keras; 
print(f'TensorFlow: {tf.__version__}'); 
print(f'Keras: {keras.__version__}')
"
