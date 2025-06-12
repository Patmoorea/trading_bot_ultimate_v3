#!/bin/bash
echo "=== STRUCTURE DES DOSSIERS ==="
tree
echo -e "\n=== CONTENU DE test_services.py ==="
cat tests/trading_services/test_services.py
echo -e "\n=== VERIFICATION DES __init__.py ==="
find . -name "__init__.py" -type f
echo -e "\n=== CONTENU DU PYTHONPATH ==="
python3 -c "import sys; print('\n'.join(sys.path))"
