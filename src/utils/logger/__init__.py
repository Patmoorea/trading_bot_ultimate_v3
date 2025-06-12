import logging
from datetime import datetime
import os

def get_logger(name: str = None) -> logging.Logger:
    """
    Crée et retourne un logger configuré
    
    Args:
        name: Nom du logger (optionnel)
        
    Returns:
        Logger configuré
    """
    # Utiliser le nom fourni ou __main__
    logger_name = name or "trading_bot"
    logger = logging.getLogger(logger_name)
    
    # Ne configurer que si pas déjà fait
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Formatter pour les logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Handler fichier
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        file_handler = logging.FileHandler(
            os.path.join(
                logs_dir,
                f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
            )
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
