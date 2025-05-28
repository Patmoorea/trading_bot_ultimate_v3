"""
API REST pour le bot de trading
Fournit un accès aux données des opportunités d'arbitrage et aux statistiques de performance
Created: 2025-05-23 04:20:00
@author: Patmooreaoui
"""

from flask import Flask, request, jsonify, Blueprint
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

from modules.database.db_manager import DatabaseManager

# Créer un Blueprint Flask pour l'API
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Gestionnaire de base de données
db_manager = None

def init_api(app: Flask, database_path: str = "data/trading_bot.db"):
    """
    Initialise l'API REST
    
    Args:
        app: Application Flask
        database_path: Chemin vers la base de données
    """
    global db_manager
    
    # Initialiser le gestionnaire de base de données
    db_manager = DatabaseManager(database_path)
    
    # Enregistrer le Blueprint
    app.register_blueprint(api_bp)
    
    logger = logging.getLogger(__name__)
    logger.info("API REST initialisée")

@api_bp.route('/opportunities', methods=['GET'])
def get_opportunities():
    """Récupère les opportunités d'arbitrage selon des critères"""
    try:
        # Récupérer les paramètres
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        arb_type = request.args.get('type')
        min_profit = request.args.get('min_profit', type=float)
        status = request.args.get('status')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        exchange = request.args.get('exchange')
        
        # Récupérer les opportunités
        opportunities = db_manager.get_opportunities(
            limit=limit,
            offset=offset,
            arb_type=arb_type,
            min_profit=min_profit,
            status=status,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange
        )
        
        return jsonify({
            'status': 'success',
            'count': len(opportunities),
            'opportunities': opportunities
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/opportunities/<int:opportunity_id>', methods=['GET'])
def get_opportunity(opportunity_id: int):
    """Récupère une opportunité par son ID"""
    try:
        opportunity = db_manager.get_opportunity_by_id(opportunity_id)
        
        if opportunity:
            return jsonify({
                'status': 'success',
                'opportunity': opportunity
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f"Opportunité #{opportunity_id} non trouvée"
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/stats/daily', methods=['GET'])
def get_daily_stats():
    """Récupère les statistiques quotidiennes"""
    try:
        # Récupérer les paramètres
        days = request.args.get('days', default=30, type=int)
        
        # Calculer les dates
        end_date = datetime.utcnow().isoformat()
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Récupérer les statistiques
        stats = db_manager.get_daily_stats(
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/stats/summary', methods=['GET'])
def get_stats_summary():
    """Récupère un résumé des statistiques"""
    try:
        # Récupérer les statistiques de tous les temps
        all_time_stats = db_manager.get_daily_stats()
        
        # Récupérer les statistiques des dernières 24h
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        today = datetime.utcnow().isoformat()
        daily_stats = db_manager.get_daily_stats(
            start_date=yesterday,
            end_date=today
        )
        
        # Récupérer les 10 meilleures opportunités
        best_opportunities = db_manager.get_opportunities(
            limit=10,
            min_profit=0,
            order_by='profit DESC'
        )
        
        return jsonify({
            'status': 'success',
            'summary': {
                'all_time': all_time_stats,
                'daily': daily_stats,
                'best_opportunities': best_opportunities
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
