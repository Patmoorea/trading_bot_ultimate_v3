"""
Module principal de l'interface web
Created: 2025-05-23
@author: Patmoorea
"""

import os
from flask import Flask, session
from flask_socketio import SocketIO
from datetime import datetime, timedelta

# Initialisation de Socket.IO pour les mises à jour en temps réel
socketio = SocketIO()

def create_app(config=None):
    """
    Crée et configure l'application Flask
    
    Args:
        config: Configuration de l'application
    
    Returns:
        Application Flask configurée
    """
    app = Flask(__name__, 
                static_folder="static",
                template_folder="templates")
    
    # Configuration par défaut
    app.config.update(
        SECRET_KEY="dev_key_change_me",
        SESSION_TYPE="filesystem",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
        TEMPLATES_AUTO_RELOAD=True
    )
    
    # Mise à jour avec la configuration fournie
    if config:
        app.config["SECRET_KEY"] = config.get("server", {}).get("secret_key", app.config["SECRET_KEY"])
        session_minutes = config.get("auth", {}).get("session_timeout_minutes", 120)
        app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=session_minutes)
    
    # Enregistrement des blueprints (routes)
    from .routes.dashboard import dashboard_bp
    from .routes.trading import trading_bp
    from .routes.settings import settings_bp
    from .routes.auth import auth_bp
    from .routes.api import api_bp
    
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Initialisation de Socket.IO avec l'application
    socketio.init_app(app, cors_allowed_origins="*")
    
    # Fonction pour vérifier l'activité du bot
    @app.context_processor
    def utility_processor():
        def bot_status():
            # À remplacer par une vérification réelle du statut du bot
            return {"status": "active", "last_update": datetime.now().strftime("%H:%M:%S")}
        return {"bot_status": bot_status}
    
    return app
