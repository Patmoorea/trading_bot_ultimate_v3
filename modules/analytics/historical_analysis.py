"""
Module d'analyse historique des données de trading
Created: 2025-05-23 04:45:00
@author: Patmoorea
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

class HistoricalAnalysis:
    """
    Analyse des données historiques pour identifier les tendances et 
    les opportunités d'arbitrage récurrentes
    """
    
    def __init__(
        self, 
        data_dir: str = 'data/historical', 
        exchanges: Optional[List[str]] = None
    ):
        """
        Initialise l'analyseur de données historiques
        
        Args:
            data_dir: Répertoire contenant les données historiques
            exchanges: Liste des exchanges à analyser
        """
        self.data_dir = data_dir
        self.exchanges = exchanges or ['binance', 'kraken', 'kucoin', 'coinbase']
        self._ensure_data_dir()
        self.logger = logging.getLogger(__name__)
        
    def _ensure_data_dir(self) -> None:
        """Crée le répertoire de données s'il n'existe pas"""
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_historical_data(
        self, 
        exchange: str, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Charge les données historiques depuis un fichier
        
        Args:
            exchange: Nom de l'exchange
            symbol: Symbole de la paire (e.g., 'BTC/USDT')
            start_date: Date de début (optionnelle)
            end_date: Date de fin (optionnelle)
            
        Returns:
            DataFrame pandas contenant les données historiques
        """
        # Sanitize symbol for filename
        file_symbol = symbol.replace('/', '_')
        filename = f"{exchange}_{file_symbol}_historical.csv"
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # Convertir la colonne timestamp en datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Appliquer le filtre de date si spécifié
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
                
                return df
            else:
                self.logger.warning(f"Fichier {file_path} introuvable. Retour d'un DataFrame vide.")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données historiques: {str(e)}")
            return pd.DataFrame()
    
    def generate_volatility_report(
        self, 
        exchange: str, 
        symbol: str, 
        window: int = 24,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Génère un rapport sur la volatilité d'une paire
        
        Args:
            exchange: Nom de l'exchange
            symbol: Symbole de la paire
            window: Fenêtre en heures pour le calcul de la volatilité
            start_date: Date de début (optionnelle)
            end_date: Date de fin (optionnelle)
            
        Returns:
            Dictionnaire contenant les métriques de volatilité
        """
        df = self.load_historical_data(exchange, symbol, start_date, end_date)
        
        if df.empty:
            return {
                'status': 'error',
                'message': f"Pas de données disponibles pour {symbol} sur {exchange}",
                'metrics': {}
            }
        
        # Calculer les rendements
        df['returns'] = df['close'].pct_change()
        
        # Calculer la volatilité
        df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(window)
        
        # Calculer les métriques
        metrics = {
            'mean_volatility': df['volatility'].mean(),
            'max_volatility': df['volatility'].max(),
            'min_volatility': df['volatility'].min(),
            'current_volatility': df['volatility'].iloc[-1],
            'volatility_trend': 'increasing' if df['volatility'].iloc[-1] > df['volatility'].iloc[-window] else 'decreasing'
        }
        
        # Identifier les périodes de haute volatilité
        high_vol_threshold = df['volatility'].quantile(0.75)
        high_vol_periods = df[df['volatility'] >= high_vol_threshold]
        
        # Analyser les périodes de la journée
        df['hour'] = df['timestamp'].dt.hour
        hourly_volatility = df.groupby('hour')['volatility'].mean().to_dict()
        
        # Ajouter au rapport
        metrics['high_volatility_threshold'] = high_vol_threshold
        metrics['high_volatility_periods_count'] = len(high_vol_periods)
        metrics['hourly_volatility'] = hourly_volatility
        metrics['best_trading_hours'] = sorted(
            hourly_volatility.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return {
            'status': 'success',
            'symbol': symbol,
            'exchange': exchange,
            'window_hours': window,
            'date_range': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'metrics': metrics
        }
    
    def find_arbitrage_patterns(
        self, 
        days_lookback: int = 30
    ) -> Dict[str, Any]:
        """
        Identifie les modèles récurrents d'opportunités d'arbitrage
        
        Args:
            days_lookback: Nombre de jours à analyser en arrière
            
        Returns:
            Dictionnaire contenant les modèles d'arbitrage identifiés
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_lookback)
        
        # Charger les opportunités d'arbitrage enregistrées
        triangular_file = os.path.join(self.data_dir, 'triangular_opportunities.csv')
        inter_exchange_file = os.path.join(self.data_dir, 'inter_exchange_opportunities.csv')
        
        if not os.path.exists(triangular_file) or not os.path.exists(inter_exchange_file):
            return {
                'status': 'error',
                'message': "Données d'opportunités d'arbitrage introuvables"
            }
        
        try:
            # Charger les données d'arbitrage triangulaire
            tri_df = pd.read_csv(triangular_file)
            tri_df['timestamp'] = pd.to_datetime(tri_df['timestamp'])
            tri_df = tri_df[(tri_df['timestamp'] >= start_date) & (tri_df['timestamp'] <= end_date)]
            
            # Charger les données d'arbitrage inter-exchanges
            inter_df = pd.read_csv(inter_exchange_file)
            inter_df['timestamp'] = pd.to_datetime(inter_df['timestamp'])
            inter_df = inter_df[(inter_df['timestamp'] >= start_date) & (inter_df['timestamp'] <= end_date)]
            
            # Analyse horaire pour l'arbitrage triangulaire
            tri_df['hour'] = tri_df['timestamp'].dt.hour
            tri_hourly = tri_df.groupby('hour').agg({
                'profit': ['mean', 'max', 'count'],
                'executed': 'sum'
            })
            tri_hourly.columns = ['avg_profit', 'max_profit', 'opportunities', 'executed']
            tri_hourly['execution_rate'] = tri_hourly['executed'] / tri_hourly['opportunities']
            
            # Meilleurs heures pour l'arbitrage triangulaire
            best_tri_hours = tri_hourly.sort_values('avg_profit', ascending=False).head(5)
            
            # Analyse horaire pour l'arbitrage inter-exchanges
            inter_df['hour'] = inter_df['timestamp'].dt.hour
            inter_hourly = inter_df.groupby('hour').agg({
                'profit': ['mean', 'max', 'count'],
                'executed': 'sum'
            })
            inter_hourly.columns = ['avg_profit', 'max_profit', 'opportunities', 'executed']
            inter_hourly['execution_rate'] = inter_hourly['executed'] / inter_hourly['opportunities']
            
            # Meilleurs heures pour l'arbitrage inter-exchanges
            best_inter_hours = inter_hourly.sort_values('avg_profit', ascending=False).head(5)
            
            # Analyse des exchanges pour l'arbitrage inter-exchanges
            best_routes = inter_df.groupby(['buy_exchange', 'sell_exchange']).agg({
                'profit': ['mean', 'max', 'count'],
                'executed': 'sum'
            })
            best_routes.columns = ['avg_profit', 'max_profit', 'opportunities', 'executed']
            best_routes['execution_rate'] = best_routes['executed'] / best_routes['opportunities']
            best_routes = best_routes.sort_values('avg_profit', ascending=False).head(5)
            
            # Créer le rapport
            report = {
                'status': 'success',
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                },
                'triangular': {
                    'total_opportunities': len(tri_df),
                    'total_executed': tri_df['executed'].sum(),
                    'avg_profit': tri_df['profit'].mean(),
                    'best_hours': best_tri_hours.to_dict(),
                    'best_exchanges': tri_df.groupby('exchange')['profit'].mean().sort_values(ascending=False).head(3).to_dict(),
                    'best_pairs': tri_df.groupby('path')['profit'].mean().sort_values(ascending=False).head(5).to_dict() if 'path' in tri_df.columns else {}
                },
                'inter_exchange': {
                    'total_opportunities': len(inter_df),
                    'total_executed': inter_df['executed'].sum(),
                    'avg_profit': inter_df['profit'].mean(),
                    'best_hours': best_inter_hours.to_dict(),
                    'best_routes': best_routes.to_dict() if not best_routes.empty else {},
                    'best_symbols': inter_df.groupby('symbol')['profit'].mean().sort_values(ascending=False).head(5).to_dict() if 'symbol' in inter_df.columns else {}
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des modèles d'arbitrage: {str(e)}")
            return {
                'status': 'error',
                'message': f"Erreur lors de l'analyse: {str(e)}"
            }
    
    def plot_volatility_heatmap(
        self,
        exchange: str,
        symbol: str,
        days_lookback: int = 30,
        save_path: Optional[str] = None
    ) -> str:
        """
        Crée une heatmap de volatilité par heure et jour de la semaine
        
        Args:
            exchange: Nom de l'exchange
            symbol: Symbole de la paire
            days_lookback: Nombre de jours à analyser en arrière
            save_path: Chemin où sauvegarder le graphique (optionnel)
            
        Returns:
            Chemin vers le graphique sauvegardé ou message d'erreur
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_lookback)
        
        df = self.load_historical_data(exchange, symbol, start_date, end_date)
        
        if df.empty:
            return f"Erreur: Pas de données disponibles pour {symbol} sur {exchange}"
        
        # Extraire l'heure et le jour de la semaine
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Calculer la volatilité horaire
        df['returns'] = df['close'].pct_change()
        volatility = df.groupby(['day_of_week', 'hour'])['returns'].std().unstack()
        
        # Créer la heatmap
        plt.figure(figsize=(12, 8))
        plt.title(f'Volatilité de {symbol} sur {exchange} par heure et jour de la semaine')
        
        days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        
        # Gérer le cas où certains jours/heures n'ont pas de données
        volatility = volatility.fillna(0)
        
        # Créer la heatmap
        heatmap = plt.pcolormesh(volatility.columns, range(len(days)), volatility.values, cmap='YlOrRd')
        
        plt.colorbar(heatmap, label='Volatilité')
        plt.xlabel('Heure de la journée')
        plt.ylabel('Jour de la semaine')
        plt.yticks(range(len(days)), days)
        plt.tight_layout()
        
        # Sauvegarder ou afficher
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            save_path = os.path.join(self.data_dir, f"{exchange}_{symbol.replace('/', '_')}_volatility_heatmap.png")
            plt.savefig(save_path)
            plt.close()
            return save_path

    def get_best_arbitrage_combinations(
        self, 
        top_n: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identifie les meilleures combinaisons pour l'arbitrage basé sur les données historiques
        
        Args:
            top_n: Nombre de meilleures combinaisons à retourner
            
        Returns:
            Dictionnaire contenant les meilleures combinaisons pour chaque type d'arbitrage
        """
        # Fichiers de données d'opportunités
        triangular_file = os.path.join(self.data_dir, 'triangular_opportunities.csv')
        inter_exchange_file = os.path.join(self.data_dir, 'inter_exchange_opportunities.csv')
        
        result = {
            'triangular': [],
            'inter_exchange': []
        }
        
        try:
            # Analyse de l'arbitrage triangulaire
            if os.path.exists(triangular_file):
                tri_df = pd.read_csv(triangular_file)
                
                if not tri_df.empty and 'path' in tri_df.columns and 'exchange' in tri_df.columns:
                    # Grouper par combinaison exchange/path
                    tri_combos = tri_df.groupby(['exchange', 'path']).agg({
                        'profit': ['mean', 'max', 'count'],
                        'executed': 'sum'
                    })
                    
                    tri_combos.columns = ['avg_profit', 'max_profit', 'appearances', 'successful_executions']
                    
                    # Calculer le taux de réussite
                    tri_combos['success_rate'] = tri_combos['successful_executions'] / tri_combos['appearances']
                    
                    # Calculer le score (combinaison de profit moyen et taux de réussite)
                    tri_combos['score'] = tri_combos['avg_profit'] * tri_combos['success_rate']
                    
                    # Trier par score et prendre les top_n
                    top_tri = tri_combos.sort_values('score', ascending=False).head(top_n)
                    
                    # Convertir en liste de dictionnaires
                    for (exchange, path), row in top_tri.iterrows():
                        result['triangular'].append({
                            'exchange': exchange,
                            'path': path if isinstance(path, str) else '->'.join(path),
                            'avg_profit': float(row['avg_profit']),
                            'max_profit': float(row['max_profit']),
                            'appearances': int(row['appearances']),
                            'success_rate': float(row['success_rate']),
                            'score': float(row['score'])
                        })
            
            # Analyse de l'arbitrage inter-exchanges
            if os.path.exists(inter_exchange_file):
                inter_df = pd.read_csv(inter_exchange_file)
                
                if not inter_df.empty and all(col in inter_df.columns for col in ['buy_exchange', 'sell_exchange', 'symbol']):
                    # Grouper par combinaison exchange/symbol
                    inter_combos = inter_df.groupby(['buy_exchange', 'sell_exchange', 'symbol']).agg({
                        'profit': ['mean', 'max', 'count'],
                        'executed': 'sum'
                    })
                    
                    inter_combos.columns = ['avg_profit', 'max_profit', 'appearances', 'successful_executions']
                    
                    # Calculer le taux de réussite
                    inter_combos['success_rate'] = inter_combos['successful_executions'] / inter_combos['appearances']
                    
                    # Calculer le score
                    inter_combos['score'] = inter_combos['avg_profit'] * inter_combos['success_rate']
                    
                    # Trier par score et prendre les top_n
                    top_inter = inter_combos.sort_values('score', ascending=False).head(top_n)
                    
                    # Convertir en liste de dictionnaires
                    for (buy_exchange, sell_exchange, symbol), row in top_inter.iterrows():
                        result['inter_exchange'].append({
                            'buy_exchange': buy_exchange,
                            'sell_exchange': sell_exchange,
                            'symbol': symbol,
                            'avg_profit': float(row['avg_profit']),
                            'max_profit': float(row['max_profit']),
                            'appearances': int(row['appearances']),
                            'success_rate': float(row['success_rate']),
                            'score': float(row['score'])
                        })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des meilleures combinaisons: {str(e)}")
            return {
                'triangular': [],
                'inter_exchange': [],
                'error': str(e)
            }
