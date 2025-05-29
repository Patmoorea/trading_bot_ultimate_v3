"""
Routes pour le dashboard principal
"""
from flask import Blueprint, render_template, current_app, request, jsonify
from datetime import datetime
import json

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
def index():
    """Page d'accueil / Dashboard"""
    return render_template('dashboard.html', 
                          title="Dashboard",
                          last_update=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dashboard_bp.route('/api/dashboard/stats')
def dashboard_stats():
    """API pour obtenir les statistiques du dashboard en temps réel"""
    # Exemple de données - à remplacer par des données réelles
    stats = {
        "portfolio_value": 12563.45,
        "daily_change_percent": 2.34,
        "open_positions": 3,
        "arbitrage_opportunities": 2,
        "last_update": datetime.now().strftime("%H:%M:%S"),
        "performance": {
            "daily": 2.34,
            "weekly": -1.23,
            "monthly": 5.67,
            "yearly": 23.45
        }
    }
    return jsonify(stats)
