#!/bin/bash
python3 -c "
from transformers import BertForSequenceClassification, BertTokenizer
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
model.save_pretrained('./model_cache/finbert')
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
tokenizer.save_pretrained('./model_cache/finbert')
print('Modèle FinBERT enregistré dans model_cache/')
"
