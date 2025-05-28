"""
Gestionnaire de configuration pour le bot de trading
Created: 2025-05-23 05:30:00
@author: Patmoorea
"""

import os
import json
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """Gestionnaire de configuration pour les différents modules du bot"""
    
    def __init__(self, config_path: str):
        """
        Initialise le gestionnaire de configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier
        
        Returns:
            Dictionnaire de configuration
        """
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Fichier de configuration '{self.config_path}' introuvable. Utilisation des valeurs par défaut.")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Erreur de parsing du fichier de configuration: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    def get_config(self) -> Dict[str, Any]:
        """
        Récupère la configuration complète
        
        Returns:
            Dictionnaire de configuration
        """
        return self.config
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur spécifique de la configuration
        
        Args:
            key: Clé de la valeur à récupérer
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur de la configuration
        """
        return self.config.get(key, default)
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Modifie une valeur dans la configuration
        
        Args:
            key: Clé de la valeur à modifier
            value: Nouvelle valeur
        """
        self.config[key] = value
    
    def save_config(self) -> bool:
        """
        Sauvegarde la configuration dans le fichier
        
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            # Créer le répertoire parent s'il n'existe pas
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Configuration sauvegardée dans '{self.config_path}'")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False
