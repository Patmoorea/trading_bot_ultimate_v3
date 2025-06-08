#!/bin/bash
mkdir -p model_cache
python3 -c "
from transformers import BertForSequenceClassification, BertTokenizer
print('Téléchargement du modèle FinBERT...')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
model.save_pretrained('./model_cache/finbert')
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
tokenizer.save_pretrained('./model_cache/finbert')
print('✓ Modèle enregistré dans model_cache/finbert')
"
