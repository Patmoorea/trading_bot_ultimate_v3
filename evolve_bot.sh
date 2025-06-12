#!/bin/zsh
# Script d'évolution contrôlée
echo "🔍 Vérification de l'état actuel..."
python -m pytest tests/ || { echo "❌ Tests échoués"; exit 1; }

echo "🔄 Application des évolutions..."
for file in evolutions/*.py; do
    echo "Applying $file"
    python "$file" || echo "⚠️ Échec partiel sur $file"
done

echo "🧪 Post-vérification..."
python -m pytest tests/ && echo "✅ Évolution réussie" || echo "❌ Problèmes détectés"
