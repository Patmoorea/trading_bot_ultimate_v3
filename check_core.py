import sys
def check_import(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False

core_modules = ['ccxt', 'pandas', 'numpy']
missing = [m for m in core_modules if not check_import(m)]

if missing:
    print("ERREUR: Modules manquants:", ", ".join(missing))
    sys.exit(1)
else:
    print("Tous les modules de base sont disponibles")
