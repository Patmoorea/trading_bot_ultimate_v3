#!/bin/bash
# Clean environment
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +

# Run tests
echo "=== RUNNING TESTS ==="
pytest tests/core/test_technical.py -v --cov=src.analysis.technical

# Manual verification
echo "\n=== MANUAL VERIFICATION ==="
python -c "
from src.analysis.technical import TechnicalAnalyzer
ta = TechnicalAnalyzer()

print('All gains (6 periods):', ta.calculate_rsi([100, 101, 102, 103, 104, 105]))
print('All losses (6 periods):', ta.calculate_rsi([105, 104, 103, 102, 101, 100]))
print('Single gain:', ta.calculate_rsi([100, 101]))
print('Single loss:', ta.calculate_rsi([101, 100]))
print('No changes:', ta.calculate_rsi([100]*10))
"
