import os
from dotenv import load_dotenv

load_dotenv()

okx_keys = {
    'OKX_API_KEY': os.getenv('OKX_API_KEY'),
    'OKX_API_SECRET': os.getenv('OKX_API_SECRET'),
    'OKX_API_PASSWORD': os.getenv('OKX_API_PASSWORD')
}

for key, value in okx_keys.items():
    if value:
        print(f"{key} est configuré : {'*' * len(value)}")
    else:
        print(f"{key} n'est PAS configuré !")
