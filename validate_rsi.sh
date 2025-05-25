#!/bin/bash
# Nettoyage complet
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +

# Installation des dépendances
pip install numpy pytest pytest-cov >/dev/null 2>&1

# Exécution des tests
echo "=== RUNNING TESTS ==="
pytest tests/core/test_technical.py -v --cov=src.analysis.technical

# Vérification manuelle compacte
echo -e "\n=== VÉRIFICATION MANUELLE ==="
python - <<END
import numpy as np
from src.analysis.technical import TechnicalAnalyzer

ta = TechnicalAnalyzer()
print('Hausse continue:', ta.calculate_rsi(np.linspace(100, 200, 50)))
print('Baisse continue:', ta.calculate_rsi(np.linspace(200, 100, 50)))
print('Stable:', ta.calculate_rsi(np.full(50, 150)))
print('Gain unique:', ta.calculate_rsi([100, 101]))
print('Perte unique:', ta.calculate_rsi([101, 100]))
END

echo -e "\n=== RUNNING TESTS ==="
pytest tests/core/test_technical.py -v --tb=short

echo -e "\n=== VÉRIFICATION MANUELLE ==="
python - <<END
import numpy as np
from src.analysis.technical import TechnicalAnalyzer

ta = TechnicalAnalyzer()
print('Hausse continue:', ta.calculate_rsi(np.linspace(100, 200, 50)))
print('Baisse continue:', ta.calculate_rsi(np.linspace(200, 100, 50)))
print('Stable:', ta.calculate_rsi(np.full(50, 150)))
print('Gain unique:', ta.calculate_rsi([100, 101]))
print('Perte unique:', ta.calculate_rsi([101, 100]))
END

set -x  # active le mode debug (affiche chaque commande)

# Lance pytest avec timeout pour éviter blocage infini (10 secondes max)
timeout 10 pytest tests/core/test_technical.py -v --tb=short || echo "Pytest timeout ou erreur"

echo -e "\n=== VÉRIFICATION MANUELLE ==="
python - <<END
import numpy as np
from src.analysis.technical import TechnicalAnalyzer

ta = TechnicalAnalyzer()
print('Hausse continue:', ta.calculate_rsi(np.linspace(100, 200, 50)))
print('Baisse continue:', ta.calculate_rsi(np.linspace(200, 100, 50)))
print('Stable:', ta.calculate_rsi(np.full(50, 150)))
print('Gain unique:', ta.calculate_rsi([100, 101]))
print('Perte unique:', ta.calculate_rsi([101, 100]))
END

echo "\n=== VÉRIFICATION RSI v2 ==="
python -c "
from src.analysis.technical import TechnicalAnalyzer
ta = TechnicalAnalyzer()

if hasattr(ta, 'calculate_rsi_v2'):
    print('1. Hausse pure:', ta.calculate_rsi_v2([100, 101, 102, 103]))
    print('2. Baisse pure:', ta.calculate_rsi_v2([103, 102, 101, 100]))
    print('3. Stable:', ta.calculate_rsi_v2([100]*10))
    print('4. Gain unique:', ta.calculate_rsi_v2([100, 101]))
    print('5. Perte unique:', ta.calculate_rsi_v2([101, 100]))
else:
    print('calculate_rsi_v2 non disponible')
"
