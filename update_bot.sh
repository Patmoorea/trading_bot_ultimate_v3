#!/bin/zsh
# Mise à jour sécurisée du bot
source venv_trading/bin/activate

echo "🔄 Mise à jour des composants..."
cat new_features/*.py >> core/  # Ajout incrémental

echo "🧪 Lancement des tests..."
python -m pytest tests/ -v --cov=core

echo "🚀 Redémarrage des services..."
sudo systemctl restart tradingbot.service

echo "✅ Mise à jour terminée. Vérifiez le dashboard:"
echo "http://localhost:8501"
