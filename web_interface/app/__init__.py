from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    
    # Configuration
    from .config import Config
    app.config.from_object(Config)
    
    # Initialisation des extensions
    socketio.init_app(app)
    
    # Enregistrement des routes
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
