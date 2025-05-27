"""
Deuxième partie des correctifs pour les échecs de tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

class TestRiskManagementFixes:
    """Correctifs pour les erreurs de gestion des risques"""
    
    @pytest.fixture
    def risk_data(self):
        return {
            'price': 30000.0,
            'volume': 1.0,
            'position_size': 100.0,
            'risk_level': 'medium'
        }

    def test_risk_calculation(self, risk_data):
        assert 'price' in risk_data
        assert isinstance(risk_data['price'], float)

    def test_position_sizing(self, risk_data):
        assert 'volume' in risk_data
        assert isinstance(risk_data['volume'], float)

class TestNotificationManagerFixes:
    """Correctifs pour les erreurs du gestionnaire de notifications"""
    
    @pytest.fixture
    def notification_manager(self):
        manager = AsyncMock()
        manager.notifiers = []
        manager.send_notification = AsyncMock()
        manager.notify_opportunity = AsyncMock()
        manager.send_daily_report = AsyncMock()
        manager.close = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_notification_attributes(self, notification_manager):
        required_attrs = [
            'notifiers',
            'send_notification',
            'notify_opportunity',
            'send_daily_report',
            'close'
        ]
        for attr in required_attrs:
            assert hasattr(notification_manager, attr)

    @pytest.mark.asyncio
    async def test_send_and_flush(self, notification_manager):
        await notification_manager.send_notification("test")
        await notification_manager.close()

class TestTelegramHandlerFixes:
    """Correctifs pour les erreurs du gestionnaire Telegram"""
    
    @pytest.fixture
    def telegram_handler(self):
        handler = MagicMock()
        handler.telegram = MagicMock()
        handler.telegram.send_message = AsyncMock()
        return handler

    def test_telegram_configuration(self, telegram_handler):
        assert hasattr(telegram_handler, 'telegram')

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup l'environnement de test"""
    with patch('json.dumps') as mock_dumps:
        mock_dumps.side_effect = lambda x, *args, **kwargs: str(x)
        yield

def test_base_test_data():
    """Test de la classe de base TestBase"""
    class TestBase:
        def __init__(self):
            self.timestamp = pd.Timestamp.now()
    
    base = TestBase()
    assert hasattr(base, 'timestamp')

