import os
from dotenv import load_dotenv

load_dotenv()

print("=== Vérification Complète ===")
print(f"Binance API Key: {'***'+os.getenv('BINANCE_API_KEY')[-3:] if os.getenv('BINANCE_API_KEY') else 'MANQUANT'}")
print(f"Telegram Token: {'***'+os.getenv('TELEGRAM_BOT_TOKEN')[-3:] if os.getenv('TELEGRAM_BOT_TOKEN') else 'MANQUANT'}")
print(f"Seuil Arbitrage: {os.getenv('ARBITRAGE_THRESHOLD', '0.3')}%")

if not os.getenv('BINANCE_API_KEY'):
    print("\nERREUR: La clé API Binance n'est pas chargée")
    print("Vérifiez que:")
    print("1. Le fichier .env existe dans le dossier du projet")
    print("2. Les variables sont correctement orthographiées")
