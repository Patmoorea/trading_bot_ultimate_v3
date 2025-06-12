#!/bin/bash
# Surveillance continue de la couverture
while true; do
    clear
    pytest --cov=src --cov-report=term-missing -v tests/unit/
    coverage html
    inotifywait -r -e modify src/ tests/ > /dev/null 2>&1
done
