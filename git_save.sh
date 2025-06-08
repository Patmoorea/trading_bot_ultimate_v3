#!/bin/bash
# Vérification préalable
if [[ $(git diff --stat) == '' ]]; then
    echo "⚠️ Aucun changement à sauvegarder"
    exit 0
fi

# Génération version
LAST_VER=$(git tag -l "v*" | sort -V | tail -1 | sed 's/v//')
NEXT_VER=${LAST_VER:-0}
NEXT_VER=$((NEXT_VER + 1))
TAG="v${NEXT_VER}_$(date +%Y%m%d_%H%M%S)"

# Opérations Git
git add .
git commit -m "🚀 ${TAG} - Optimisation logging/imports"
git tag -a "${TAG}" -m "Backup ${TAG}"

# Push avec vérification
if git push origin main && git push --tags; then
    echo -e "\n✅ Sauvegarde réussie"
    echo "🔖 Version: ${TAG}"
    echo "📌 Commit: $(git rev-parse --short HEAD)"
else
    echo "❌ Erreur pendant git push"
    git reset --soft HEAD~1
    git tag -d "${TAG}"
fi
