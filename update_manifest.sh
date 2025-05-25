#!/bin/bash

# Configuration des modules critiques
CRITICAL_MODULES=(
  "news_analysis"
  "adaptive_stoploss" 
  "usdc_arbitrage"
  "telegram_integration"
  "order_optimization"
)

# Génération du manifeste enrichi
{
  echo '{'
  echo '  "project_id": "trading_bot_ultimate",'
  echo '  "last_scan": "'$(date -u +'%Y-%m-%dT%H:%M:%SZ')'",'
  
  # Scan des fichiers normaux
  echo '  "file_map": {'
  find src -type f -printf '    "%p": {"size": %s, "mtime": "%TY-%Tm-%TdT%TTZ"},\n' | sed '$ s/,$//'
  echo '  },'
  
  # Vérification spéciale des modules critiques
  echo '  "critical_modules": {'
  for mod in "${CRITICAL_MODULES[@]}"; do
    status=$(find src -type d -name "*$mod*" | wc -l)
    echo "    \"$mod\": $status,"
  done
  echo '    "checked_at": "'$(date -u +'%Y-%m-%dT%H:%M:%SZ')'"'
  echo '  },'
  
  # Checksum de sécurité
  echo '  "checksum": "'$(find src -type f -exec sha256sum {} + | sort | sha256sum | cut -d' ' -f1)'"'
  echo '}'
} > .bot_manifest.json

echo "✅ Manifeste généré avec vérification des modules critiques"
