#!/bin/bash
cd /Users/patricejourdan/trading_bot_ultimate
python -m pytest --cov=src.core --cov-report=html
open htmlcov/index.html
