#!/bin/bash

# 1. Clean up
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# 2. Patch config imports
for file in $(find tests/ -name "*.py"); do
    sed -i '' 's/from src.config import Config/from src.core.config import Config/g' "$file"
done

# 3. Disable numba in test mode
echo "Config.USE_NUMBA = False" >> src/core/__init__.py

# 4. Run basic tests
pytest tests/unit/core/test_{basic,import,model}.py -v
