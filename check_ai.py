ai_modules = ['tensorflow', 'sklearn']
missing_ai = [m for m in ai_modules if not check_import(m)]

if missing_ai:
    print("Avertissement: Modules IA manquants:", ", ".join(missing_ai))
    print("Pour installer: pip install", " ".join(missing_ai))
else:
    print("Tous les modules IA sont disponibles")
