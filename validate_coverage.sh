#!/bin/bash
# Nettoyage
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +

# Installation
pip install pytest-cov >/dev/null 2>&1

# Tests avec couverture ciblée
echo "=== LANCEMENT DES TESTS ==="
pytest tests/valid/ \
  --cov=src \
  --cov-config=.coveragerc \
  --cov-report=term-missing

# Génération du rapport HTML
echo "=== GENERATION DU RAPPORT ==="
pytest --cov=src --cov-report=html

# Vérification finale
echo "=== VERIFICATION MANUELLE ==="
python -c "
from src.data.stream_manager import StreamManager
from src.analysis.technical import TechnicalAnalyzer
from src.news.sentiment import NewsSentimentAnalyzer
from src.execution.order_executor import OrderExecutor

print('StreamManager:', StreamManager().buffer_size > 0)
print('TechnicalAnalyzer:', TechnicalAnalyzer().calculate_rsi([1,2,3]) != 0)
print('NewsAnalyzer:', 'positive' in NewsSentimentAnalyzer().sentiment_pipeline('test'))
print('OrderExecutor:', OrderExecutor().execute_order({})['status'] == 'filled')
"
