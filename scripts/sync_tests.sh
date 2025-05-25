#!/bin/bash
# Synchronisation bidirectionnelle
rsync -auv --include='test_*.py' --exclude='*' ./ tests/unit/
rsync -auv --include='test_*.py' --exclude='*' tests/unit/ ./
