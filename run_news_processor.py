#!/usr/bin/env python3
import sys
from pathlib import Path

# Ajout du chemin racine au PYTHONPATH
root_path = str(Path(__file__).parent)
sys.path.append(root_path)

if __name__ == "__main__":
    print("🚀 Démarrage du processeur de news...")
    try:
        from src.news_processor.main import main
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Arrêt du processeur de news...")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        raise
