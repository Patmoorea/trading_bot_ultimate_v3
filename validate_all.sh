#!/bin/bash
# Clean environment
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +

# Run all tests
echo "=== RUNNING ALL TESTS ==="
pytest tests/ --cov=src --cov-report=term-missing

# Manual verification
echo "\n=== MANUAL RSI VERIFICATION ==="
python -c "
from src.analysis.technical import TechnicalAnalyzer
ta = TechnicalAnalyzer()
print('All gains:', ta.calculate_rsi([100, 101, 102, 103]))
print('All losses:', ta.calculate_rsi([103, 102, 101, 100]))
print('Alternating:', ta.calculate_rsi([100, 101]*20))
"
