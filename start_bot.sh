#!/bin/zsh

# Activation environnement
source .venv/bin/activate
[ -f .env ] && source .env

# Démarrer les services
python src/data_collection/main.py &
sleep 2

python src/ai_engine/decision_engine.py &
sleep 1

python src/execution/order_executor.py &
sleep 1

streamlit run src/monitoring/dashboard.py
