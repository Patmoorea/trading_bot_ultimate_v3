## Migration vers UnifiedArbitrage

### Anciens Modules Intégrés
- `simple_arbitrage.py` → Méthode cross-exchange
- `advanced_arbitrage.py` → Gestion des configurations
- `temporal_arbitrage.py` → Détection temporelle
- `arbitrage_engine.py` → Compatibilité préservée

### Nouveautés
- Gestion centralisée des configurations
- Logging unifié
- Typage fort avec dataclasses
- Meilleure gestion des erreurs

### Procédure
1. Mettre à jour les imports pour utiliser `core.py`
2. Utiliser `compat.py` pour la rétrocompatibilité
3. Migrer progressivement les scripts
