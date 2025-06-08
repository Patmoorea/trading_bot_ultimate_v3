#!/usr/bin/env python3
"""
Connecteur de données pour l'interface web
Créé: 2025-05-23
Auteur: Patmoorea
"""
import json
import logging
import threading
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Configuration du logging
logger = logging.getLogger(__name__)

class WebDataConnector:
    """
    Connecteur de données pour l'interface web.
    Fournit les données depuis les différents modules du bot
    vers l'interface web.
    """
    
    def __init__(self):
        """Initialisation du connecteur"""
        self.cache = {}
        self.cache_expiry = {}
        self.update_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Démarrer le thread de mise à jour
        self.start_update_thread()
    
    def start_update_thread(self):
        """Démarre le thread de mise à jour des données"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Thread de mise à jour des données démarré")
    
    def stop_update_thread(self):
        """Arrête le thread de mise à jour des données"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
            logger.info("Thread de mise à jour des données arrêté")
    
    def _update_loop(self):
        """Boucle de mise à jour des données en arrière-plan"""
        while self.running:
            try:
                # Mise à jour des données d'arbitrage
                self._update_arbitrage_data()
                
                # Mise à jour des données de performance
                self._update_performance_data()
                
                # Intervalle de mise à jour (5 secondes)
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de mise à jour: {e}")
                time.sleep(10)
    
    def _update_arbitrage_data(self):
        """Récupère les données d'arbitrage depuis les modules du bot"""
        try:
            # Import ici pour éviter les imports circulaires
            from src.strategies.arbitrage.multi_exchange.multi_arbitrage import MultiExchangeArbitrage
            
            # Récupérer les opportunités d'arbitrage
            arbitrage = MultiExchangeArbitrage()
            opportunities = arbitrage.check_arbitrage()
            
            # Mise à jour du cache avec verrou
            with self.lock:
                self.cache['arbitrage_opportunities'] = opportunities
                self.cache_expiry['arbitrage_opportunities'] = datetime.now() + timedelta(seconds=30)
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données d'arbitrage: {e}")
    
    def _update_performance_data(self):
        """Récupère les données de performance depuis les modules du bot"""
        try:
            # Import ici pour éviter les imports circulaires
            from modules.analytics.performance_analyzer import PerformanceAnalyzer
            
            # Récupérer les métriques de performance
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.get_metrics()
            
            # Mise à jour du cache avec verrou
            with self.lock:
                self.cache['performance_metrics'] = metrics
                self.cache_expiry['performance_metrics'] = datetime.now() + timedelta(minutes=5)
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données de performance: {e}")
    
    def get_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """
        Récupère les opportunités d'arbitrage actuelles
        
        Returns:
            Liste des opportunités d'arbitrage
        """
        # Vérifier si les données sont en cache et valides
        with self.lock:
            if ('arbitrage_opportunities' in self.cache and 
                'arbitrage_opportunities' in self.cache_expiry and 
                datetime.now() < self.cache_expiry['arbitrage_opportunities']):
                return self.cache['arbitrage_opportunities']
        
        # Si pas en cache, forcer la mise à jour
        self._update_arbitrage_data()
        
        # Retourner les données
        with self.lock:
            return self.cache.get('arbitrage_opportunities', [])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de performance actuelles
        
        Returns:
            Dictionnaire des métriques de performance
        """
        # Vérifier si les données sont en cache et valides
        with self.lock:
            if ('performance_metrics' in self.cache and 
                'performance_metrics' in self.cache_expiry and 
                datetime.now() < self.cache_expiry['performance_metrics']):
                return self.cache['performance_metrics']
        
        # Si pas en cache, forcer la mise à jour
        self._update_performance_data()
        
        # Retourner les données
        with self.lock:
            return self.cache.get('performance_metrics', {})

# Test du module en standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    connector = WebDataConnector()
    print("Web connector initialized. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        connector.stop_update_thread()
        print("Connector stopped.")
