import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_vars():
    # Chemin vers le fichier .env dans le dossier parent
    env_path = Path('/Users/patricejourdan/Desktop/trading_bot_ultimate/.env')
    
    # Charger les variables d'environnement
    load_dotenv(dotenv_path=env_path)
    
    return {
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
    }
