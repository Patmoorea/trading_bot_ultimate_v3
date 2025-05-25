#!/bin/bash

# Clean pyc files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# Create necessary directories
mkdir -p logs/performance
mkdir -p models
mkdir -p src/analysis/technical
mkdir -p src/core
mkdir -p src/ai
mkdir -p src/data

# Set environment variables for tests
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
export IS_TEST=true
export TELEGRAM_BOT_TOKEN=test_token
export TELEGRAM_CHAT_ID=test_chat_id
export EXCHANGE_API_KEY=test_key
export EXCHANGE_API_SECRET=test_secret
export MODEL_PATH=models/
export PERFORMANCE_LOG_PATH=logs/performance/

# Run tests
echo "Running all tests..."
python -m pytest tests_new/ -v -s --import-mode=append
