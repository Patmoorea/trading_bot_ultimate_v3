#!/bin/bash
# 1. Nettoyage
pip uninstall -y tensorflow keras tensorboard

# 2. Installation spécifique
pip install tensorflow==2.15.0

# 3. Vérification
python -c "
try:
    from tensorflow import keras
    print('SUCCÈS: Keras importé (version', keras.__version__ + ')')
except ImportError as e:
    print('ÉCHEC:', str(e))
"

# 4. Exécution des tests
if python -c "import tensorflow" &> /dev/null; then
    python -m pytest tests/ai/test_hybrid.py -v
else
    echo "TensorFlow n'est toujours pas installé correctement"
fi
