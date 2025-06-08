"""
Test Extensions for adding common functionality to test classes
Version 1.0.0 - Created: 2025-05-26 06:04:12 by Patmoorea
"""

import pandas as pd
import numpy as np
from tests_new.base_test import BaseTest

def add_test_data_support():
    """Decorator to add get_test_data method to test classes"""
    def decorator(cls):
        if not hasattr(cls, 'get_test_data'):
            def get_test_data(self, symbol: str = "BTC/USDT", *args, **kwargs):
                """Get test data for unit tests"""
                dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
                data = {
                    'timestamp': dates,
                    'open': np.random.uniform(30000, 35000, 100),
                    'high': np.random.uniform(31000, 36000, 100),
                    'low': np.random.uniform(29000, 34000, 100),
                    'close': np.random.uniform(30000, 35000, 100),
                    'volume': np.random.uniform(100, 1000, 100)
                }
                return pd.DataFrame(data)
            cls.get_test_data = get_test_data
        return cls
    return decorator
"""
Test Extensions
Version 1.0.0 - Created: 2025-05-26 06:10:38 by Patmoorea
"""

from tests_new.base_test import with_test_data

def apply_test_data(test_module):
    """Apply test data decorator to all test classes in a module"""
    for item in dir(test_module):
        if item.startswith('Test'):
            cls = getattr(test_module, item)
            if isinstance(cls, type):
                setattr(test_module, item, with_test_data(cls))
    return test_module
"""
Test Extensions - Adapte les tests au code existant
Version 1.0.0 - Created: 2025-05-26 17:26:35 by Patmoorea
"""

from functools import wraps
import pandas as pd
import numpy as np
from datetime import datetime

def adapt_test_data(cls):
    """
    Décorateur qui adapte les tests aux méthodes existantes
    sans modifier les fichiers originaux
    """
    original_setup = getattr(cls, 'setUp', None)
    
    def new_setup(self):
        """Setup étendu qui préserve le setup original"""
        if original_setup:
            original_setup(self)
            
        # Ajouter get_test_data si nécessaire
        if not hasattr(self, 'get_test_data'):
            def get_test_data(self, *args, **kwargs):
                """Compatibilité avec l'API existante"""
                dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
                data = {
                    'timestamp': dates,
                    'open': np.random.uniform(30000, 35000, 100),
                    'high': np.random.uniform(31000, 36000, 100),
                    'low': np.random.uniform(29000, 34000, 100),
                    'close': np.random.uniform(30000, 35000, 100),
                    'volume': np.random.uniform(100, 1000, 100)
                }
                return pd.DataFrame(data)
            setattr(self.__class__, 'get_test_data', get_test_data)
            
        # Ajouter timestamp si nécessaire
        if not hasattr(self, 'timestamp'):
            self.timestamp = "2025-05-26 17:26:35"
            
        # Ajouter user si nécessaire
        if not hasattr(self, 'user'):
            self.user = "Patmoorea"
    
    cls.setUp = new_setup
    return cls

# Application automatique aux classes de test
import sys
for module_name in list(sys.modules.keys()):
    if module_name.startswith('tests_new.unit'):
        module = sys.modules[module_name]
        for item_name in dir(module):
            if item_name.startswith('Test'):
                item = getattr(module, item_name)
                if isinstance(item, type):
                    setattr(module, item_name, adapt_test_data(item))
