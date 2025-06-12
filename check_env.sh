#!/bin/bash
REQUIRED_MODULES=("dotenv" "ccxt" "asyncio")

for module in "${REQUIRED_MODULES[@]}"; do
    python -c "import ${module}" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installation de ${module}..."
        pip install ${module}
    fi
done

echo "Vérification terminée. Tous les modules sont installés."
