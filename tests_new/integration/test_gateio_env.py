import os
from dotenv import load_dotenv

load_dotenv()

gateio_keys = {
    'GATE_IO_API_KEY': os.getenv('GATE_IO_API_KEY'),
    'GATE_IO_API_SECRET': os.getenv('GATE_IO_API_SECRET')
}

print("\nTest des variables d'environnement Gate.io :")
for key, value in gateio_keys.items():
    if value:
        print(f"{key} est configuré : {'*' * len(value)}")
    else:
        print(f"{key} n'est PAS configuré !")
