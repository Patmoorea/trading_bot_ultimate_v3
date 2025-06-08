import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestPerformance:
    @pytest.fixture
    def sample_data(self):
        # Créer des données de test
        dates = pd.date_range(
            start=datetime(2025, 5, 27, 7, 17, 14) - timedelta(days=30),
            end=datetime(2025, 5, 27, 7, 17, 14),
            freq='1H'
        )
        
        np.random.seed(42)  # Pour la reproductibilité
        
        data = pd.DataFrame({
            'timestamp': dates,
            'price': np.random.normal(30000, 1000, len(dates)),
            'volume': np.random.lognormal(10, 1, len(dates)),
            'trades': np.random.randint(10, 100, len(dates))
        })
        
        data.set_index('timestamp', inplace=True)
        return data

    @pytest.fixture
    def performance_metrics(self):
        return {
            'total_trades': 100,
            'winning_trades': 60,
            'losing_trades': 40,
            'total_profit': 5000.0,
            'max_drawdown': -1000.0,
            'sharpe_ratio': 1.5,
            'win_rate': 0.60
        }

    def test_performance_logging(self, sample_data, tmp_path):
        # Créer un fichier de log temporaire
        log_file = tmp_path / "performance.log"
        
        # Enregistrer les données de performance
        with open(log_file, 'w') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            for col in sample_data.columns:
                f.write(f"{col}_mean: {sample_data[col].mean():.2f}\n")
        
        # Vérifier que le fichier existe et contient des données
        assert log_file.exists()
        content = log_file.read_text()
        assert "Timestamp:" in content
        assert "price_mean:" in content
        assert "volume_mean:" in content
        assert "trades_mean:" in content

    def test_performance_calculation(self, sample_data, performance_metrics):
        # Calculer les métriques de performance
        results = self.calculate_performance_metrics(sample_data)
        
        # Vérifier les métriques calculées
        assert isinstance(results, dict)
        assert all(metric in results for metric in [
            'total_trades',
            'winning_trades',
            'losing_trades',
            'total_profit',
            'max_drawdown',
            'sharpe_ratio',
            'win_rate'
        ])
        
        # Vérifier que les valeurs sont dans des plages raisonnables
        assert 0 <= results['win_rate'] <= 1
        assert results['sharpe_ratio'] > -10 and results['sharpe_ratio'] < 10
        assert isinstance(results['total_trades'], (int, np.integer))
        assert results['winning_trades'] + results['losing_trades'] == results['total_trades']

    def calculate_performance_metrics(self, data):
        """Calcule les métriques de performance à partir des données"""
        if data.empty:
            return None
            
        # Calculer les rendements
        returns = data['price'].pct_change().dropna()
        
        # Simuler quelques trades
        n_trades = 100
        trade_results = np.random.choice([1, -1], n_trades, p=[0.6, 0.4])
        trade_profits = trade_results * np.abs(np.random.normal(50, 20, n_trades))
        
        winning_trades = sum(trade_results > 0)
        losing_trades = sum(trade_results < 0)
        
        return {
            'total_trades': n_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_profit': float(sum(trade_profits)),
            'max_drawdown': float(min(np.minimum.accumulate(returns.cumsum()))),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)),
            'win_rate': float(winning_trades / n_trades)
        }
