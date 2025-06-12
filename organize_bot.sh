#!/bin/bash

# Migration sécurisée (ne déplace que les fichiers non existants)
safe_mv() {
    if [ ! -e "$2" ]; then
        mv "$1" "$2"
        echo "Déplacé: $1 -> $2"
    else
        echo "Conflit: $2 existe déjà (conservé)"
    fi
}

# Réorganisation des modules
mkdir -p core/strategies/
safe_mv modules/arbitrage/ core/strategies/
safe_mv modules/trading/ core/strategies/
safe_mv modules/risk/ core/management/

# Migration des utilitaires
mkdir -p libs/utils/
safe_mv modules/utils/ libs/utils/

# Organisation des configurations
mkdir -p configs/environments/
safe_mv config.py configs/environments/production.py
safe_mv .env configs/environments/development.env

echo "Réorganisation terminée. Vérifiez les conflits éventuels."
