#!/bin/bash
BASE_DIR="/Users/patricejourdan/trading_bot_ultimate"

find "$BASE_DIR/src" -type f -name "*.py" | while read file; do
    rel_path=${file#$BASE_DIR/src/}
    test_path="$BASE_DIR/tests/unit/${rel_path%.py}.py"
    
    mkdir -p "$(dirname "$test_path")"
    
    if [[ ! -f "$test_path" ]]; then
        module_name=$(echo "${rel_path%.py}" | tr '/' '.')
        cat > "$test_path" << EOF
import pytest
from src.$module_name import *

class Test${module_name##*.}:
    @pytest.fixture
    def fixture(self):
        return # Initialiser le module ici
    
    def test_feature(self, fixture):
        # Template de test
        assert True
EOF
    fi
done
