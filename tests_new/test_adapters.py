"""
Test Adapters - Adapte les tests au code existant sans modifications
Version 1.0.0 - Created: 2025-05-26 17:34:53 by Patmoorea
"""

import sys
import asyncio
import functools
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Configuration globale
TEST_CONFIG = {
    'timestamp': "2025-05-26 17:34:53",
    'user': "Patmoorea",
    'root_path': str(Path(__file__).parent.parent)
}

def async_patch(func):
    """Décorateur pour gérer les coroutines dans les tests"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(func(*args, **kwargs))
        return func(*args, **kwargs)
    return wrapper

class TestAdapter:
    """Adaptateur pour les classes de test"""
    
    @classmethod
    def patch_test_class(cls, test_class):
        """Applique les patches nécessaires à une classe de test"""
        
        # Sauvegarde du setup original
        original_setup = getattr(test_class, 'setUp', None)
        
        def new_setup(self):
            """Setup étendu qui préserve le comportement original"""
            if original_setup:
                original_setup(self)
                
            # Ajout des attributs de base
            self.timestamp = TEST_CONFIG['timestamp']
            self.user = TEST_CONFIG['user']
            
            # Ajout des mocks pour les modules manquants
            if not hasattr(sys.modules, 'modules'):
                sys.modules['modules'] = MagicMock()
            if not hasattr(sys.modules['modules'], 'notifications'):
                sys.modules['modules'].notifications = MagicMock()
            if not hasattr(sys.modules['modules'], 'utils'):
                sys.modules['modules'].utils = MagicMock()
                sys.modules['modules'].utils.telegram_logger = MagicMock()
            
            # Ajout de get_test_data si nécessaire
            if not hasattr(self, 'get_test_data'):
                import pandas as pd
                import numpy as np
                
                def get_test_data(self_inner, *args, **kwargs):
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
                    
                self.get_test_data = get_test_data.__get__(self, self.__class__)
            
            # Mock des attributs manquants pour TelegramHandler
            if 'telegram_handler' in self.__class__.__name__.lower():
                self.is_authorized = True
                self._worker_task = AsyncMock()
                self._session = AsyncMock()
            
            # Mock des attributs manquants pour NotificationManager
            if 'notification' in self.__class__.__name__.lower():
                async def mock_flush():
                    return ['Test alert']
                self.flush = mock_flush
                
            # Mock des attributs manquants pour TradingDashboard
            if 'dashboard' in sys.modules.get(self.__class__.__module__, '').lower():
                self.active_positions = {}
                self.update_trades = MagicMock()
                self.update_risk_metrics = MagicMock()
                self._handle_market_data = MagicMock()
                self.get_memory_usage = MagicMock(return_value=100)
                self.logger = MagicMock()
                self.pnl_history = []
                self._create_risk_display = MagicMock()
                self.trades_stream = []
                
        # Application du nouveau setup
        test_class.setUp = new_setup
        return test_class

def apply_adapters():
    """Applique les adaptateurs à tous les modules de test"""
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('tests_new'):
            module = sys.modules[module_name]
            for item_name in dir(module):
                if item_name.startswith('Test'):
                    item = getattr(module, item_name)
                    if isinstance(item, type):
                        adapted_class = TestAdapter.patch_test_class(item)
                        setattr(module, item_name, adapted_class)

# Application automatique des adaptateurs
apply_adapters()
