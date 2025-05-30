import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from src.dashboard.main_dashboard import EnhancedTradingDashboard, NotificationManager, TradingMetrics

class TestDashboardIntegration(unittest.TestCase):
    def setUp(self):
        """Configuration initiale pour chaque test"""
        self.dashboard = EnhancedTradingDashboard()
        self.notifications = NotificationManager()
        
    def test_metrics_update(self):
        """Test la mise à jour des métriques"""
        test_metrics = {
            'daily_pnl': 5.5,
            'total_pnl': 15.5,
            'win_rate': 65.0,
            'drawdown': -2.5,
            'sharpe_ratio': 1.8,
            'trades_count': 42,
            'active_positions': 3
        }
        self.dashboard.update_metrics(test_metrics)
        self.assertEqual(self.dashboard.metrics.daily_pnl, 5.5)
        self.assertEqual(self.dashboard.metrics.win_rate, 65.0)
        
    def test_metrics_full_update(self):
        """Test la mise à jour complète des métriques"""
        test_metrics = {
            'daily_pnl': 5.5,
            'total_pnl': 15.5,
            'win_rate': 65.0,
            'drawdown': -2.5,
            'sharpe_ratio': 1.8,
            'trades_count': 42,
            'active_positions': 3
        }
        metrics = self.dashboard.update_metrics(test_metrics)
        
        # Vérification de tous les champs
        for key, value in test_metrics.items():
            self.assertEqual(getattr(metrics, key), value)
            
    def test_notification_system(self):
        """Test le système de notifications"""
        test_message = "Test Alert Message"
        count = self.notifications.add_alert(
            test_message,
            level="warning",
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        self.assertEqual(count, 1)
        latest_notification = self.notifications.notifications[0]
        self.assertEqual(latest_notification['message'], test_message)
        self.assertEqual(latest_notification['level'], "warning")
        
    def test_notification_expiry(self):
        """Test l'expiration des notifications"""
        past_time = datetime.utcnow() - timedelta(hours=2)
        future_time = datetime.utcnow() + timedelta(hours=2)
        
        # Ajout de notifications avec différentes dates d'expiration
        self.notifications.add_alert("Expired", expiry=past_time)
        self.notifications.add_alert("Not expired", expiry=future_time)
        
        # Vérification des notifications actives
        active_alerts = self.notifications.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0]['message'], "Not expired")

if __name__ == '__main__':
    unittest.main(verbosity=2)
