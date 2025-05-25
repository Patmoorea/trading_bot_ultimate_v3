class SmartRouter:
    """Version simplifiée sans dépendance Binance"""
    def __init__(self, config):
        self.config = config
        self.exchanges = {}  # Dictionnaire pour gérer différents exchanges
    
    def add_exchange(self, name, adapter):
        """Ajoute un connecteur d'exchange"""
        self.exchanges[name] = adapter
