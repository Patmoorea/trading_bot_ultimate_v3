"""
Module d'intégration des données de marché avec l'analyse de performance
Created: 2025-05-23 05:25:00
@author: Patmoorea
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from modules.analytics.performance_analyzer import PerformanceAnalyzer
from modules.analytics.market_data_provider import MarketDataProvider

class MarketPerformanceAnalyzer:
    """
    Analyse la performance en fonction des conditions de marché
    """
    
    def __init__(self):
        """Initialise l'analyseur de performance basé sur les conditions de marché"""
        self.logger = logging.getLogger(__name__)
        self.performance_analyzer = PerformanceAnalyzer()
        self.market_data_provider = MarketDataProvider()
    
    def analyze_performance_by_market_conditions(self, trades_file: str = "data/trades.csv") -> Dict:
        """
        Analyse les performances en fonction des conditions de marché
        
        Args:
            trades_file: Chemin vers le fichier de transactions
            
        Returns:
            Résultats de l'analyse par conditions de marché
        """
        # Charger les transactions
        self.performance_analyzer.trades_file = trades_file
        trades_df = self.performance_analyzer._load_trades()
        
        if trades_df.empty:
            self.logger.warning("Aucune transaction à analyser.")
            return {}
        
        # Enrichir les transactions avec les données de marché
        enriched_df = self.market_data_provider.enrich_trades_with_market_data(trades_df)
        
        # Analyser par conditions de marché
        market_conditions = {}
        
        # Analyser par volatilité du marché
        if 'market_condition' in enriched_df.columns:
            volatility_analysis = enriched_df.groupby('market_condition').agg({
                'profit': ['count', 'sum', 'mean', 'std'],
                'market_volatility': 'mean'
            }).reset_index()
            
            market_conditions['volatility'] = volatility_analysis.to_dict()
        
        # Analyser par direction du marché
        if 'market_direction' in enriched_df.columns:
            direction_analysis = enriched_df.groupby('market_direction').agg({
                'profit': ['count', 'sum', 'mean', 'std'],
                'market_trend': 'mean'
            }).reset_index()
            
            market_conditions['direction'] = direction_analysis.to_dict()
        
        # Analyser par stratégie et conditions de marché
        if 'strategy' in enriched_df.columns and 'market_condition' in enriched_df.columns:
            strategy_condition_analysis = enriched_df.groupby(['strategy', 'market_condition']).agg({
                'profit': ['count', 'sum', 'mean', 'std']
            }).reset_index()
            
            market_conditions['strategy_by_condition'] = strategy_condition_analysis.to_dict()
        
        # Analyser par stratégie et direction du marché
        if 'strategy' in enriched_df.columns and 'market_direction' in enriched_df.columns:
            strategy_direction_analysis = enriched_df.groupby(['strategy', 'market_direction']).agg({
                'profit': ['count', 'sum', 'mean', 'std']
            }).reset_index()
            
            market_conditions['strategy_by_direction'] = strategy_direction_analysis.to_dict()
        
        return market_conditions
    
    def generate_market_performance_report(self, trades_file: str = "data/trades.csv", save_path: str = "data/reports") -> Dict:
        """
        Génère un rapport complet de performance par conditions de marché
        
        Args:
            trades_file: Chemin vers le fichier de transactions
            save_path: Chemin où sauvegarder le rapport
            
        Returns:
            Résultats de l'analyse
        """
        # Analyser les performances par conditions de marché
        market_analysis = self.analyze_performance_by_market_conditions(trades_file)
        
        # Obtenir les métriques générales
        general_metrics = self.performance_analyzer.calculate_metrics()
        
        # Combiner les rapports
        report = {
            'general_metrics': general_metrics,
            'market_analysis': market_analysis,
            'timestamp': datetime.now().isoformat(),
            'report_type': 'market_performance'
        }
        
        # Sauvegarder le rapport
        import os
        import json
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            report_file = os.path.join(save_path, f'market_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Rapport sauvegardé dans {report_file}")
        
        return report
    
    def find_optimal_market_conditions(self, trades_file: str = "data/trades.csv") -> Dict:
        """
        Identifie les conditions de marché optimales pour chaque stratégie
        
        Args:
            trades_file: Chemin vers le fichier de transactions
            
        Returns:
            Conditions optimales par stratégie
        """
        # Charger les transactions
        self.performance_analyzer.trades_file = trades_file
        trades_df = self.performance_analyzer._load_trades()
        
        if trades_df.empty:
            self.logger.warning("Aucune transaction à analyser.")
            return {}
        
        # Enrichir les transactions avec les données de marché
        enriched_df = self.market_data_provider.enrich_trades_with_market_data(trades_df)
        
        if 'strategy' not in enriched_df.columns:
            self.logger.warning("Colonne 'strategy' manquante dans les transactions.")
            return {}
        
        optimal_conditions = {}
        
        for strategy in enriched_df['strategy'].unique():
            strategy_df = enriched_df[enriched_df['strategy'] == strategy]
            
            # Conditions de marché optimales
            if 'market_condition' in strategy_df.columns:
                volatility_perf = strategy_df.groupby('market_condition')['profit'].mean()
                best_volatility = volatility_perf.idxmax() if not volatility_perf.empty else None
            else:
                best_volatility = None
            
            # Direction de marché optimale
            if 'market_direction' in strategy_df.columns:
                direction_perf = strategy_df.groupby('market_direction')['profit'].mean()
                best_direction = direction_perf.idxmax() if not direction_perf.empty else None
            else:
                best_direction = None
            
            optimal_conditions[strategy] = {
                'best_volatility': best_volatility,
                'best_direction': best_direction,
                'avg_profit_in_optimal_conditions': strategy_df[
                    (strategy_df['market_condition'] == best_volatility if best_volatility else True) & 
                    (strategy_df['market_direction'] == best_direction if best_direction else True)
                ]['profit'].mean() if not strategy_df.empty else 0
            }
        
        return optimal_conditions
