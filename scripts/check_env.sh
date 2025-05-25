#!/bin/bash

echo "=== Variables requises ==="
echo "NEWS_API_KEY: $(grep NEWS_API_KEY .env || echo 'Non trouvée')"
echo "FINBERT_ENABLED: $(grep FINBERT_ENABLED .env || echo 'false')"

echo "=== Test de lecture ==="
python3 -c "
from dotenv import load_dotenv; load_dotenv()
import os
print('NEWS_API_KEY:', os.getenv('NEWS_API_KEY', 'Non définie'))
"
