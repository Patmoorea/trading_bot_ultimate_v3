# Trading Bot Ultimate v3

Bot de trading avancé développé par Patmoorea, intégrant analyse technique, IA et news.

## Fonctionnalités

### 1. Collecte de Données Ultra-Performante
- 12 flux WebSocket simultanés
- Buffer circulaire (~15ms latence)
- Compression LZ4

### 2. Analyse Multi-Timeframe
- 42 indicateurs techniques
- Fusion multi-timeframe
- Patterns avancés

### 3. Intelligence Artificielle
- CNN-LSTM hybride
- Optimisation Metal (M1/M2/M4)
- Transfer Learning

### 4. Gestion des Risques
- Stop-loss dynamique
- Circuit breakers
- Protection anti-snipe

### 5. News & Sentiment
- 12 sources d'actualités
- Analyse NLP temps réel
- Impact score

## Installation

\```bash
# 1. Cloner le repo
git clone https://github.com/Patmoorea/trading_bot_ultimate_v3.git
cd trading_bot_ultimate_v3

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env
# Éditer .env avec vos clés API

# 4. Lancer le bot
python main.py
\```

## Configuration

Le fichier .env doit contenir :
- BINANCE_API_KEY
- BINANCE_API_SECRET
- TELEGRAM_BOT_TOKEN
- NEWS_API_KEYS

## Usage

1. Configuration des paires :
\```python
pairs = ['BTCUSDT', 'ETHUSDT']
timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
\```

2. Paramètres de risque :
\```python
risk_params = {
    'max_drawdown': 0.05,    # 5%
    'daily_stop': 0.02,      # 2%
    'position_size': 'auto'  # Kelly Criterion
}
\```

## Tests

\```bash
pytest tests/
\```

## Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push sur la branche
5. Créer une Pull Request

## Licence

MIT License
