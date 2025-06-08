"""
Dashboard web pour le bot de trading
Fournit une interface graphique pour visualiser les données et les performances
Created: 2025-05-23 04:25:00
@author: Patmooreaoui
"""

from flask import Flask, request, jsonify, render_template, Blueprint
import logging
import os
from typing import Dict

from modules.database.db_manager import DatabaseManager
from modules.dashboard.api import init_api

# Créer un Blueprint Flask pour le dashboard
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/')

# Gestionnaire de base de données
db_manager = None

def create_app(config: Dict = None):
    """
    Crée et configure l'application Flask
    
    Args:
        config: Configuration du dashboard
    
    Returns:
        Application Flask configurée
    """
    app = Flask(__name__, 
                static_folder='static',
                template_folder='templates')
    
    # Configurer l'application
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'trading_bot_secret')
    
    if config:
        app.config.update(config)
    
    database_path = config.get('database_path', 'data/trading_bot.db') if config else 'data/trading_bot.db'
    
    # Initialiser l'API
    init_api(app, database_path)
    
    # Initialiser le gestionnaire de base de données pour le dashboard
    global db_manager
    db_manager = DatabaseManager(database_path)
    
    # Enregistrer le Blueprint
    app.register_blueprint(dashboard_bp)
    
    logger = logging.getLogger(__name__)
    logger.info("Application Dashboard initialisée")
    
    return app

@dashboard_bp.route('/')
def index():
    """Page d'accueil du dashboard"""
    return render_template('index.html')

@dashboard_bp.route('/opportunities')
def opportunities_view():
    """Page de visualisation des opportunités"""
    arb_type = request.args.get('type')
    min_profit = request.args.get('min_profit', type=float)
    
    return render_template('opportunities.html', 
                         arb_type=arb_type,
                         min_profit=min_profit)

@dashboard_bp.route('/stats')
def stats_view():
    """Page de statistiques"""
    days = request.args.get('days', default=30, type=int)
    
    return render_template('stats.html', days=days)

@dashboard_bp.route('/settings')
def settings_view():
    """Page de configuration"""
    return render_template('settings.html')

def run_dashboard(host='0.0.0.0', port=5000, debug=False, config=None):
    """
    Exécute le serveur du dashboard
    
    Args:
        host: Hôte du serveur
        port: Port du serveur
        debug: Mode debug
        config: Configuration du dashboard
    """
    app = create_app(config)
    app.run(host=host, port=port, debug=debug)
