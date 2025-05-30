import logging
from flask_socketio import emit

class WebSocketManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connected = False
        
    def handle_connect(self):
        self.connected = True
        self.logger.info("WebSocket connected")
        
    def handle_disconnect(self):
        self.connected = False
        self.logger.info("WebSocket disconnected")
        
    def send_market_data(self, data):
        if self.connected:
            emit('market_data', data)
