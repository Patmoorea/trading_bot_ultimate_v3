#!/bin/bash

echo "=== Vérification des imports ==="
for file in $(find . -name "*.py"); do
    if grep -q "from.*sentiment" "$file"; then
        echo "Import dans $file :"
        grep "from.*sentiment" "$file"
    fi
done

echo "=== Test des fonctionnalités ==="
python3 -c "
from core.news.sentiment import NewsAnalyzer
print('Core analyzer:', NewsAnalyzer().analyze('Test')[0][0])
from src.modules.news.sentiment import get_enhanced_sentiment
print('Enhanced:', get_enhanced_sentiment('Test'))
"
