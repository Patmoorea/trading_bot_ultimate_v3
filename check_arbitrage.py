import os

def check_config():
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'ARBITRAGE_THRESHOLD'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"ERREUR: Configuration manquante: {missing}")
        print("Assurez-vous que :")
        print("1. Le fichier .env existe")
        print("2. Les variables sont définies")
        return False
    
    print("Configuration arbitrage OK")
    return True

if __name__ == "__main__":
    check_config()
