"""
Test Configuration and Adaptations
Version 1.0.0 - Created: 2025-05-26 17:30:34 by Patmoorea
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

# Configuration globale des tests
TEST_CONFIG = {
    'timestamp': "2025-05-26 17:30:34",
    'user': "Patmoorea",
    'base_path': str(Path(__file__).parent)
}

def adapt_test_class(cls):
    """Adaptateur de classe de test qui préserve le comportement existant"""
    
    # Sauvegarde des méthodes originales
    original_setup = getattr(cls, 'setUp', None)
    
    def adapted_setup(self):
        """Setup adapté qui ajoute les fonctionnalités manquantes"""
        if original_setup:
            original_setup(self)
            
        # Ajout des attributs de base
        self.timestamp = TEST_CONFIG['timestamp']
        self.user = TEST_CONFIG['user']
        
        # Ajout de get_test_data si nécessaire
        if not hasattr(self, 'get_test_data'):
            def get_test_data(self, *args, **kwargs):
                dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
                return pd.DataFrame({
                    'timestamp': dates,
                    'open': np.random.uniform(30000, 35000, 100),
                    'high': np.random.uniform(31000, 36000, 100),
                    'low': np.random.uniform(29000, 34000, 100),
                    'close': np.random.uniform(30000, 35000, 100),
                    'volume': np.random.uniform(100, 1000, 100)
                })
            self.get_test_data = get_test_data.__get__(self)
            
        # Mock pour les tests de notifications
        if 'notifications' in cls.__module__:
            def async_mock():
                mock = MagicMock()
                mock.__aenter__ = MagicMock()
                mock.__aexit__ = MagicMock()
                return mock
            self._session = async_mock()
            self.is_authorized = True
            self._worker_task = MagicMock()
    
    # Application des adaptations
    cls.setUp = adapted_setup
    
    return cls

def patch_modules():
    """Crée les liens symboliques nécessaires sans modifier les fichiers"""
    
    base_path = Path(TEST_CONFIG['base_path']).parent
    modules_path = base_path / 'modules'
    
    # Crée les répertoires nécessaires
    os.makedirs(str(modules_path / 'utils'), exist_ok=True)
    
    # Crée les liens symboliques si nécessaire
    if not (modules_path / 'utils' / 'telegram_logger.py').exists():
        src = base_path / 'src/modules/utils/telegram_logger.py'
        dst = modules_path / 'utils' / 'telegram_logger.py'
        if src.exists() and not dst.exists():
            os.symlink(str(src), str(dst))

# Application automatique des adaptations
def auto_adapt_tests():
    """Applique automatiquement les adaptations aux classes de test"""
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('tests_new'):
            module = sys.modules[module_name]
            for item_name in dir(module):
                if item_name.startswith('Test'):
                    item = getattr(module, item_name)
                    if isinstance(item, type):
                        adapted_item = adapt_test_class(item)
                        setattr(module, item_name, adapted_item)

# Création des liens symboliques au démarrage
patch_modules()
