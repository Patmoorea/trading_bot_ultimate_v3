#!/bin/bash
MODULES=(
    "risk_management"
    "engine" 
    "ai_engine"
)

for module in "${MODULES[@]}"; do
    echo "Testing $module..."
    pytest tests/unit/core/test_${module}.py -v --cov=src.core.${module}
done
