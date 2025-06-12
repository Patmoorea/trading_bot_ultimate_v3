#!/bin/zsh
echo "⚡ Installation du Trading Bot Ultimate ⚡"

# 1. Création de l'environnement
python -m venv .venv
source .venv/bin/activate

# 2. Installation des dépendances principales
pip install --upgrade pip
pip install -r requirements.txt

# 3. Installation des optimisations M4
if [[ "$(uname -m)" == "arm64" ]]; then
    echo "🔧 Installation des optimisations Apple Silicon..."
    pip install tensorflow-macos tensorflow-metal
    pip install numba
fi

# 4. Initialisation des dossiers
mkdir -p data/{market,signals,backtest,news}
mkdir -p logs

echo "✅ Installation terminée!"
echo "Pour commencer:"
echo "1. Configurez le fichier .env"
echo "2. Lancez les tests: pytest"
echo "3. Démarrez le bot: python src/core/engine.py"
