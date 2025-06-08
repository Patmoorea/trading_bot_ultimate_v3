import importlib

required = ['ccxt', 'tensorflow', 'pandas', 'numpy', 'pytest', 'asyncio', 'sklearn']

print("=== État des dépendances ===")
for lib in required:
    try:
        importlib.import_module(lib)
        print(f"\033[92m✓ {lib}\033[0m")
    except ImportError:
        print(f"\033[91m✗ {lib} (manquant)\033[0m")

print("\nPour installer les dépendances manquantes :")
print("pip install", " ".join([lib for lib in required]))
