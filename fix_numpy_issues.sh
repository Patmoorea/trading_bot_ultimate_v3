#!/bin/bash
# Correction des problèmes NumPy 2.x

echo "1. Downgrade de NumPy..."
pip install "numpy<2" --force-reinstall

echo "2. Réinstallation des dépendances..."
pip install --upgrade --force-reinstall     tensorflow     tensorboard     stable-baselines3     gymnasium     "protobuf<4"

echo "3. Application des correctifs..."
cat <<EOL > tf_compat_fix.py
import numpy as np
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
EOL

echo "4. Vérification..."
python -c "import numpy as np; import tensorflow as tf; print('NumPy:', np.__version__, 'TF:', tf.__version__)"

echo "Correction terminée. Exécutez les tests avec:"
echo "python -m pytest tests/ai/test_hybrid.py -v"
