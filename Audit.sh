#!/bin/sh

# Exclusion spécifique du manifeste pendant sa création
if git diff --cached --name-only | grep -q '.bot_manifest.json'; then
  exit 0
fi

# Vérification adaptée
MISSING=$(jq -r '
  .critical_modules | 
  to_entries[] | 
  select(.value == 0) | 
  .key
' .bot_manifest.json | tr '\n' ' ')

if [ -n "$MISSING" ]; then
  echo "⚠️ Modules manquants : $MISSING"
  echo "Pour contourner : git commit -n (no-verify)"
  exit 1
fi
