import requests
import hmac
import hashlib
import time
import os
import json
from dotenv import load_dotenv

def test_binance_api():
    # Chargement des clés
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    # Paramètres
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp,
        'recvWindow': 5000
    }

    # Signature
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    params['signature'] = signature

    # Requête
    url = "https://api.binance.com/api/v3/account"
    headers = {'X-MBX-APIKEY': api_key}

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            
            # Afficher uniquement les balances non nulles
            print("\n=== Balances non nulles ===")
            for balance in data['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:
                    print(f"{balance['asset']}: Free={free}, Locked={locked}")
            
            print(f"\nPermissions: {data.get('permissions', [])}")
            print(f"Account Status: Active")
            return True
        else:
            print(f"Erreur {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Test de connexion Binance")
    test_binance_api()
