#!/bin/bash

# Mise à jour des dépendances
pip install -U \
    tensorflow-macos \
    tensorflow-metal \
    ccxt \
    websockets \
    lz4 \
    pyarrow \
    optuna \
    stable-baselines3 \
    transformers

# Installation des dépendances custom
git clone https://github.com/Patmoorea/trading_bot_ultimate.git
cd trading_bot_ultimate
python setup.py develop

# Configuration
cp config/.env.example config/.env
echo "Please edit config/.env with your API keys"
