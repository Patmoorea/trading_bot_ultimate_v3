#!/bin/bash
# Outils d'analyse
pip install \
    pytest-cov \
    hyperfine \
    pip-audit \
    radon \
    safety \
    pylint

# Dépendances projet
pip install \
    TA-Lib \
    transformers \
    torch \
    sentencepiece \
    python-binance \
    ccxt
