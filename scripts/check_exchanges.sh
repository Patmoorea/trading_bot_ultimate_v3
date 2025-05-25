#!/bin/bash
# Script de vérification des exchanges - Updated: 2025-05-17 23:03:03
# @author: Patmoorea

echo "Vérification des connexions aux exchanges..."

exchanges=("binance" "gateio" "bingx" "okx" "blofin")

for exchange in "${exchanges[@]}"; do
    echo "Testing $exchange connection..."
    python3 -c "
import ccxt
import os
from dotenv import load_dotenv
load_dotenv()
try:
    exchange = ccxt.${exchange}({
        'apiKey': os.getenv('${exchange^^}_API_KEY'),
        'secret': os.getenv('${exchange^^}_API_SECRET')
    })
    print(f'✅ {exchange.name} connection successful')
except Exception as e:
    print(f'❌ {exchange.name} connection failed: {str(e)}')
"
done
