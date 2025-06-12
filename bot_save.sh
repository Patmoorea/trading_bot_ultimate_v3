#!/bin/bash
PYTHON_CMD="python3"
MSG="${1:-Auto-save via script}"

$PYTHON_CMD -c "
from src.utils.autosave import GitAutoSaver
print('✅ Backup successful' if GitAutoSaver().create_snapshot('$MSG') else '⏩ No changes to save')
"
