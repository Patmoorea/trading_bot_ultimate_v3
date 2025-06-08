"""
Module d'analyse des performances du bot de trading
Created: 2025-05-23 05:15:00
@author: Patmoorea
"""

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from typing import List, Dict, Any, Tuple, Optional

class PerformanceAnalyzer:
    """
    Analyse les performances du bot de trading et génère des rapports
    """
    
    def __init__(self, data_dir='data', log_dir='logs', report_dir='reports'):
        """
        Initialise l'analyseur de performances
        
        Args:
            data_dir: Répertoire pour les données
            log_dir: Répertoire des logs
            report_dir: Répertoire pour les rapports générés
        """
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.report_dir = report_dir
        
        # Créer les répertoires s'ils n'existent pas
        for directory in [data_dir, report_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_transaction_logs(self, log_file=None, days=None) -> pd.DataFrame:
        """
        Charge les logs de transactions dans un DataFrame
        
        Args:
            log_file: Fichier de log spécifique à charger
            days: Nombre de jours à analyser (si log_file n'est pas spécifié)
            
        Returns:
            DataFrame contenant les transactions
        """
        if log_file:
            file_path = os.path.join(self.log_dir, log_file)
        else:
            # Par défaut, utiliser le fichier de logs des transactions
            file_path = os.path.join(self.log_dir, 'trading_transactions.log')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier de log {file_path} n'existe pas")
        
        # Charger les logs au format JSON
        transactions = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Chaque ligne est un JSON
                    transaction = json.loads(line.strip())
                    transactions.append(transaction)
                except json.JSONDecodeError:
                    continue  # Ignorer les lignes mal formatées
        
        # Convertir en DataFrame
        df = pd.DataFrame(transactions)
        
        # Convertir les timestamps en datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filtrer par date si spécifié
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[df['timestamp'] >= cutoff_date]
        
        return df
    
    def load_performance_logs(self, log_file=None, days=None) -> pd.DataFrame:
        """
        Charge les logs de performances dans un DataFrame
        
        Args:
            log_file: Fichier de log spécifique à charger
            days: Nombre de jours à analyser (si log_file n'est pas spécifié)
            
        Returns:
            DataFrame contenant les métriques de performance
        """
        if log_file:
            file_path = os.path.join(self.log_dir, log_file)
        else:
            # Par défaut, utiliser le fichier de logs des performances
            file_path = os.path.join(self.log_dir, 'trading_performance.log')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Le fichier de log {file_path} n'existe pas")
        
        # Charger les logs au format JSON
        performances = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Extraire le timestamp et le JSON
                    parts = line.strip().split(' - ', 1)
                    if len(parts) != 2:
                        continue
                    
                    log_time, json_str = parts
                    
                    # Parser le JSON
                    perf_data = json.loads(json_str)
                    
                    # Ajouter le timestamp du log
                    perf_data['log_time'] = log_time
                    
                    performances.append(perf_data)
                except (json.JSONDecodeError, ValueError):
                    continue  # Ignorer les lignes mal formatées
        
        # Convertir en DataFrame
        df = pd.DataFrame(performances)
        
        # Convertir les timestamps en datetime
        for col in ['timestamp', 'log_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Filtrer par date si spécifié
        if days and 'timestamp' in df.columns:
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
        
        return df
    
    def calculate_daily_stats(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les statistiques quotidiennes à partir des transactions
        
        Args:
            transactions_df: DataFrame des transactions
            
        Returns:
            DataFrame avec les statistiques quotidiennes
        """
        if transactions_df.empty:
            return pd.DataFrame()
        
        # S'assurer que le timestamp est au format datetime
        if 'timestamp' in transactions_df.columns:
            transactions_df['date'] = transactions_df['timestamp'].dt.date
        else:
            return pd.DataFrame()
        
        # Grouper par date
        daily_stats = transactions_df.groupby('date').agg({
            'profit': ['sum', 'mean', 'min', 'max', 'count'],
            'amount': ['sum', 'mean']
        })
        
        # Aplatir les colonnes multi-index
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
        
        # Calculer les métriques supplémentaires
        if 'profit_sum' in daily_stats.columns and 'amount_sum' in daily_stats.columns:
            daily_stats['roi_daily'] = (daily_stats['profit_sum'] / daily_stats['amount_sum']) * 100
        
        # Réinitialiser l'index pour avoir la date comme colonne
        daily_stats = daily_stats.reset_index()
        
        return daily_stats
    
    def calculate_cumulative_stats(self, daily_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les statistiques cumulatives à partir des stats quotidiennes
        
        Args:
            daily_stats: DataFrame des statistiques quotidiennes
            
        Returns:
            DataFrame avec les statistiques cumulatives
        """
        if daily_stats.empty:
            return pd.DataFrame()
        
        # Copier le DataFrame
        cumulative = daily_stats.copy()
        
        # Calculer les métriques cumulatives
        for col in ['profit_sum', 'amount_sum', 'profit_count']:
            if col in cumulative.columns:
                cumulative[f'{col}_cumulative'] = cumulative[col].cumsum()
        
        # Calculer le ROI cumulatif
        if 'profit_sum_cumulative' in cumulative.columns and 'amount_sum_cumulative' in cumulative.columns:
            cumulative['roi_cumulative'] = (cumulative['profit_sum_cumulative'] / cumulative['amount_sum_cumulative']) * 100
        
        return cumulative
    
    def generate_performance_metrics(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Génère des métriques de performance complètes
        
        Args:
            transactions_df: DataFrame des transactions
            
        Returns:
            Dictionnaire de métriques
        """
        if transactions_df.empty:
            return {'error': 'Aucune transaction disponible'}
        
        metrics = {}
        
        # Période d'analyse
        if 'timestamp' in transactions_df.columns:
            metrics['period_start'] = transactions_df['timestamp'].min().isoformat()
            metrics['period_end'] = transactions_df['timestamp'].max().isoformat()
            metrics['days_analyzed'] = (transactions_df['timestamp'].max() - transactions_df['timestamp'].min()).days + 1
        
        # Métriques globales
        if 'profit' in transactions_df.columns:
            metrics['total_profit'] = float(transactions_df['profit'].sum())
            metrics['avg_profit_per_trade'] = float(transactions_df['profit'].mean())
            metrics['max_profit'] = float(transactions_df['profit'].max())
            metrics['min_profit'] = float(transactions_df['profit'].min())
        
        if 'amount' in transactions_df.columns:
            metrics['total_volume'] = float(transactions_df['amount'].sum())
            metrics['avg_trade_size'] = float(transactions_df['amount'].mean())
            
            # ROI global
            if 'profit' in transactions_df.columns:
                metrics['roi'] = float((transactions_df['profit'].sum() / transactions_df['amount'].sum()) * 100)
        
        # Métriques par type d'arbitrage
        if 'type' in transactions_df.columns:
            metrics['transactions_by_type'] = {}
            for arb_type, group in transactions_df.groupby('type'):
                metrics['transactions_by_type'][arb_type] = {
                    'count': int(len(group)),
                    'total_profit': float(group['profit'].sum()),
                    'avg_profit': float(group['profit'].mean()),
                    'success_rate': float((group['profit'] > 0).mean() * 100)
                }
        
        # Métriques de risque
        if 'profit' in transactions_df.columns:
            returns = transactions_df['profit'].pct_change().dropna()
            
            if len(returns) > 0:
                metrics['risk'] = {
                    'volatility': float(returns.std() * 100),  # En pourcentage
                    'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252))  # Annualisé
                }
                
                # Value at Risk (95%)
                metrics['risk']['var_95'] = float(np.percentile(returns, 5))
                
                # Maximum Drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / running_max - 1)
                metrics['risk']['max_drawdown'] = float(drawdown.min() * 100)  # En pourcentage
        
        # Analyse des heures de trading
        if 'timestamp' in transactions_df.columns:
            transactions_df['hour'] = transactions_df['timestamp'].dt.hour
            hourly_profits = transactions_df.groupby('hour')['profit'].sum()
            
            metrics['hourly_analysis'] = {
                'most_profitable_hour': int(hourly_profits.idxmax()),
                'least_profitable_hour': int(hourly_profits.idxmin()),
                'hourly_distribution': {str(h): float(p) for h, p in hourly_profits.items()}
            }
        
        # Taux de réussite global
        if 'profit' in transactions_df.columns:
            metrics['success_rate'] = float((transactions_df['profit'] > 0).mean() * 100)
            metrics['transaction_count'] = int(len(transactions_df))
            metrics['profitable_count'] = int((transactions_df['profit'] > 0).sum())
            metrics['losing_count'] = int((transactions_df['profit'] <= 0).sum())
        
        return metrics
    
    def plot_daily_profits(self, daily_stats: pd.DataFrame, output_file=None):
        """
        Génère un graphique des profits quotidiens
        
        Args:
            daily_stats: DataFrame des statistiques quotidiennes
            output_file: Fichier de sortie pour le graphique (optionnel)
        """
        if daily_stats.empty or 'date' not in daily_stats.columns or 'profit_sum' not in daily_stats.columns:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Tracer les profits quotidiens
        plt.bar(daily_stats['date'], daily_stats['profit_sum'], color='blue', alpha=0.7)
        
        # Ligne de tendance
        z = np.polyfit(range(len(daily_stats)), daily_stats['profit_sum'], 1)
        p = np.poly1d(z)
        plt.plot(daily_stats['date'], p(range(len(daily_stats))), "r--", linewidth=2)
        
        # Formatage
        plt.title('Profits Quotidiens', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Profit', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Rotation des dates pour une meilleure lisibilité
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            output_path = os.path.join(self.report_dir, output_file)
            plt.savefig(output_path)
        
        plt.close()
    
    def plot_cumulative_profit(self, cumulative_stats: pd.DataFrame, output_file=None):
        """
        Génère un graphique des profits cumulatifs
        
        Args:
            cumulative_stats: DataFrame des statistiques cumulatives
            output_file: Fichier de sortie pour le graphique (optionnel)
        """
        if (cumulative_stats.empty or 'date' not in cumulative_stats.columns 
            or 'profit_sum_cumulative' not in cumulative_stats.columns):
            return
        
        plt.figure(figsize=(12, 6))
        
        # Tracer les profits cumulatifs
        plt.plot(cumulative_stats['date'], cumulative_stats['profit_sum_cumulative'], 
                 color='green', linewidth=2, marker='o', markersize=4)
        
        # Remplir sous la courbe
        plt.fill_between(cumulative_stats['date'], 0, cumulative_stats['profit_sum_cumulative'], 
                          color='green', alpha=0.2)
        
        # Formatage
        plt.title('Profits Cumulatifs', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Profit Cumulatif', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Rotation des dates pour une meilleure lisibilité
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            output_path = os.path.join(self.report_dir, output_file)
            plt.savefig(output_path)
        
        plt.close()
    
    def plot_profit_distribution(self, transactions_df: pd.DataFrame, output_file=None):
        """
        Génère un histogramme de la distribution des profits
        
        Args:
            transactions_df: DataFrame des transactions
            output_file: Fichier de sortie pour le graphique (optionnel)
        """
        if transactions_df.empty or 'profit' not in transactions_df.columns:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Tracer l'histogramme
        plt.hist(transactions_df['profit'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        
        # Ligne verticale à profit = 0
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        # Formatage
        plt.title('Distribution des Profits', fontsize=16)
        plt.xlabel('Profit', fontsize=12)
        plt.ylabel('Nombre de Transactions', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if output_file:
            output_path = os.path.join(self.report_dir, output_file)
            plt.savefig(output_path)
        
        plt.close()
    
    def generate_full_report(self, days=30, output_prefix='report'):
        """
        Génère un rapport complet de performance
        
        Args:
            days: Nombre de jours à analyser
            output_prefix: Préfixe pour les fichiers de sortie
            
        Returns:
            Dictionnaire avec les statistiques et les chemins des graphiques générés
        """
        # Créer le répertoire de rapports s'il n'existe pas
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Timestamp pour les noms de fichiers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report = {
            'timestamp': timestamp,
            'period_days': days,
            'charts': {},
            'metrics': {}
        }
        
        try:
            # Charger les transactions
            transactions_df = self.load_transaction_logs(days=days)
            
            if transactions_df.empty:
                report['error'] = 'Aucune transaction disponible pour la période spécifiée'
                return report
            
            # Calculer les statistiques
            daily_stats = self.calculate_daily_stats(transactions_df)
            cumulative_stats = self.calculate_cumulative_stats(daily_stats)
            
            # Générer les métriques
            report['metrics'] = self.generate_performance_metrics(transactions_df)
            
            # Générer les graphiques
            charts_info = {
                'daily_profits': {'method': self.plot_daily_profits, 'df': daily_stats},
                'cumulative_profit': {'method': self.plot_cumulative_profit, 'df': cumulative_stats},
                'profit_distribution': {'method': self.plot_profit_distribution, 'df': transactions_df}
            }
            
            for chart_name, info in charts_info.items():
                output_file = f"{output_prefix}_{chart_name}_{timestamp}.png"
                info['method'](info['df'], output_file=output_file)
                report['charts'][chart_name] = os.path.join(self.report_dir, output_file)
            
            # Sauvegarder le rapport au format JSON
            report_file = f"{output_prefix}_{timestamp}.json"
            report_path = os.path.join(self.report_dir, report_file)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            report['report_file'] = report_path
            
        except Exception as e:
            report['error'] = str(e)
        
        return report
    
    def export_data_for_ml(self, days=90, output_file=None):
        """
        Exporte les données pour l'entraînement de modèles de machine learning
        
        Args:
            days: Nombre de jours de données à exporter
            output_file: Nom du fichier de sortie (optionnel)
            
        Returns:
            Chemin du fichier exporté
        """
        if output_file is None:
            output_file = f"ml_data_{datetime.now().strftime('%Y%m%d')}.csv"
        
        output_path = os.path.join(self.data_dir, output_file)
        
        try:
            # Charger les transactions
            transactions_df = self.load_transaction_logs(days=days)
            
            # Ajouter des fonctionnalités pour le ML
            if not transactions_df.empty and 'timestamp' in transactions_df.columns:
                # Extraire les caractéristiques temporelles
                transactions_df['hour'] = transactions_df['timestamp'].dt.hour
                transactions_df['day_of_week'] = transactions_df['timestamp'].dt.dayofweek
                transactions_df['day_of_month'] = transactions_df['timestamp'].dt.day
                transactions_df['month'] = transactions_df['timestamp'].dt.month
                
                # Sauvegarder au format CSV
                transactions_df.to_csv(output_path, index=False)
                
                return output_path
            
            return None
            
        except Exception as e:
            print(f"Erreur lors de l'export des données pour ML: {e}")
            return None
