"""
Module de journalisation avancé pour le bot de trading
Created: 2025-05-23 05:00:00
@author: Patmoorea
"""

import os
import logging
import json
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading
import traceback

class AdvancedLogger:
    """
    Logger avancé avec rotation des fichiers, formatage JSON et niveaux multiples
    """
    
    def __init__(self, name, log_dir='logs', max_size_mb=10, backup_count=10):
        """
        Initialise le logger avancé
        
        Args:
            name: Nom du logger (utilisé comme préfixe pour les fichiers)
            log_dir: Répertoire où stocker les logs
            max_size_mb: Taille maximale de chaque fichier de log en Mo
            backup_count: Nombre de fichiers de backup à conserver
        """
        self.name = name
        self.log_dir = log_dir
        self.max_size_mb = max_size_mb
        self.backup_count = backup_count
        
        # Créer le répertoire de logs s'il n'existe pas
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialiser les différents loggers
        self.info_logger = self._setup_logger(f"{name}_info", os.path.join(log_dir, f"{name}_info.log"), logging.INFO)
        self.error_logger = self._setup_logger(f"{name}_error", os.path.join(log_dir, f"{name}_error.log"), logging.ERROR)
        self.debug_logger = self._setup_logger(f"{name}_debug", os.path.join(log_dir, f"{name}_debug.log"), logging.DEBUG)
        
        # Logger pour les transactions (format JSON)
        self.transaction_logger = self._setup_json_logger(f"{name}_transactions", os.path.join(log_dir, f"{name}_transactions.log"))
        
        # Logger pour les performances (rotation horaire)
        self.performance_logger = self._setup_timed_logger(f"{name}_performance", os.path.join(log_dir, f"{name}_performance.log"))
        
        # Thread-safe lock pour l'écriture de logs
        self.lock = threading.Lock()
    
    def _setup_logger(self, name, log_file, level):
        """
        Configure un logger standard avec rotation des fichiers
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Éviter les handlers en double
        if logger.handlers:
            return logger
        
        # Handler pour la rotation des fichiers
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_size_mb * 1024 * 1024,
            backupCount=self.backup_count
        )
        
        # Formateur
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Ajouter le handler au logger
        logger.addHandler(handler)
        
        return logger
    
    def _setup_json_logger(self, name, log_file):
        """
        Configure un logger avec format JSON pour les transactions
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Éviter les handlers en double
        if logger.handlers:
            return logger
        
        # Handler pour la rotation des fichiers
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_size_mb * 1024 * 1024,
            backupCount=self.backup_count
        )
        
        # Pas de formateur standard car nous utilisons JSON
        
        # Ajouter le handler au logger
        logger.addHandler(handler)
        
        return logger
    
    def _setup_timed_logger(self, name, log_file):
        """
        Configure un logger avec rotation horaire pour les performances
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Éviter les handlers en double
        if logger.handlers:
            return logger
        
        # Handler pour la rotation horaire
        handler = TimedRotatingFileHandler(
            log_file,
            when='h',  # rotation horaire
            interval=1,
            backupCount=24 * 7  # garder une semaine de logs
        )
        
        # Formateur
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Ajouter le handler au logger
        logger.addHandler(handler)
        
        return logger
    
    def info(self, message):
        """
        Enregistre un message d'information
        """
        with self.lock:
            self.info_logger.info(message)
    
    def error(self, message, exc_info=None):
        """
        Enregistre un message d'erreur
        """
        with self.lock:
            if exc_info:
                self.error_logger.error(message, exc_info=True)
            else:
                self.error_logger.error(message)
    
    def debug(self, message):
        """
        Enregistre un message de debug
        """
        with self.lock:
            self.debug_logger.debug(message)
    
    def log_transaction(self, transaction_data):
        """
        Enregistre une transaction au format JSON
        
        Args:
            transaction_data: Dictionnaire contenant les données de la transaction
        """
        with self.lock:
            # Ajouter un timestamp
            transaction_data['timestamp'] = datetime.now().isoformat()
            
            # Convertir en JSON
            json_data = json.dumps(transaction_data)
            
            # Enregistrer
            self.transaction_logger.info(json_data)
    
    def log_performance(self, metric_name, value, context=None):
        """
        Enregistre une métrique de performance
        
        Args:
            metric_name: Nom de la métrique
            value: Valeur de la métrique
            context: Contexte supplémentaire (optionnel)
        """
        with self.lock:
            data = {
                'metric': metric_name,
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            
            if context:
                data['context'] = context
            
            # Convertir en JSON
            json_data = json.dumps(data)
            
            # Enregistrer
            self.performance_logger.info(json_data)
    
    def log_exception(self, message="Exception non gérée"):
        """
        Enregistre une exception avec sa stack trace complète
        
        Args:
            message: Message à afficher avant la stack trace
        """
        with self.lock:
            self.error_logger.error(f"{message}\n{traceback.format_exc()}")

# Classe pour mesurer les performances
class PerformanceTracker:
    """
    Utilitaire pour mesurer le temps d'exécution des fonctions
    """
    
    def __init__(self, logger):
        """
        Initialise le tracker de performance
        
        Args:
            logger: Instance de AdvancedLogger à utiliser
        """
        self.logger = logger
        self.start_times = {}
    
    def start(self, operation_name):
        """
        Démarre le chronométrage d'une opération
        
        Args:
            operation_name: Nom de l'opération à chronométrer
        """
        self.start_times[operation_name] = time.time()
    
    def stop(self, operation_name, context=None):
        """
        Arrête le chronométrage et enregistre la durée
        
        Args:
            operation_name: Nom de l'opération
            context: Contexte supplémentaire (optionnel)
            
        Returns:
            Durée de l'opération en secondes
        """
        if operation_name not in self.start_times:
            self.logger.error(f"L'opération '{operation_name}' n'a pas été démarrée")
            return None
        
        duration = time.time() - self.start_times[operation_name]
        
        # Log la performance
        self.logger.log_performance(
            f"duration_{operation_name}",
            round(duration, 6),
            context
        )
        
        # Supprimer le temps de démarrage
        del self.start_times[operation_name]
        
        return duration

# Fonction décorateur pour mesurer les performances
def measure_performance(logger, operation_name=None, log_args=False):
    """
    Décorateur pour mesurer les performances d'une fonction
    
    Args:
        logger: Instance de AdvancedLogger à utiliser
        operation_name: Nom de l'opération (par défaut, nom de la fonction)
        log_args: Si True, log les arguments de la fonction
        
    Returns:
        Décorateur
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Déterminer le nom de l'opération
            op_name = operation_name or func.__name__
            
            # Contexte pour le log
            context = None
            if log_args:
                # Limiter la taille des arguments pour éviter des logs trop volumineux
                args_str = str(args)[:100] + ('...' if len(str(args)) > 100 else '')
                kwargs_str = str(kwargs)[:100] + ('...' if len(str(kwargs)) > 100 else '')
                context = {'args': args_str, 'kwargs': kwargs_str}
            
            # Démarrer le chronomètre
            start_time = time.time()
            
            try:
                # Exécuter la fonction
                result = func(*args, **kwargs)
                
                # Calculer la durée
                duration = time.time() - start_time
                
                # Log la performance
                logger.log_performance(f"func_{op_name}", round(duration, 6), context)
                
                return result
            
            except Exception as e:
                # En cas d'erreur, loguer l'exception
                duration = time.time() - start_time
                error_context = {'duration': round(duration, 6), 'error': str(e)}
                if context:
                    error_context.update(context)
                
                logger.log_performance(f"error_{op_name}", round(duration, 6), error_context)
                logger.log_exception(f"Exception dans {op_name}")
                
                # Relancer l'exception
                raise
        
        return wrapper
    
    return decorator
