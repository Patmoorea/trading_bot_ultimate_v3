"""
API endpoints pour l'interface web du bot de trading
Created: 2025-05-23 04:30:00
@author: Patmoorea
"""

from flask import Blueprint, jsonify, request, current_app
import os
import json
import time
from datetime import datetime

# Import du mock_data pour la phase de transition
from .mock_data import get_mock_opportunities, get_mock_stats

api = Blueprint('api', __name__)

def get_data_file(file_path):
    """
    Lit un fichier JSON et retourne son contenu
    Retourne un dictionnaire vide si le fichier n'existe pas
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            current_app.logger.error(f"Erreur lors de la lecture de {file_path}: {e}")
    
    return {}

@api.route('/status')
def status():
    """Renvoie l'état du bot"""
    status_path = os.path.join(current_app.config['DATA_DIR'], 'status', 'current_status.json')
    status_data = get_data_file(status_path)
    
    if not status_data:
        # Fallback to mock data if real data isn't available
        return jsonify({
            'status': 'active',
            'uptime': 3600,  # secondes
            'last_scan': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    return jsonify(status_data.get('status', {}))

@api.route('/opportunities/triangular')
def triangular_opportunities():
    """Renvoie les dernières opportunités d'arbitrage triangulaire"""
    opps_path = os.path.join(current_app.config['DATA_DIR'], 'opportunities', 'triangular.json')
    opps_data = get_data_file(opps_path)
    
    if not opps_data:
        # Fallback to mock data if real data isn't available
        opportunities = get_mock_opportunities('triangular')
        return jsonify(opportunities)
    
    return jsonify(opps_data)

@api.route('/opportunities/inter-exchange')
def inter_exchange_opportunities():
    """Renvoie les dernières opportunités d'arbitrage inter-exchanges"""
    opps_path = os.path.join(current_app.config['DATA_DIR'], 'opportunities', 'inter_exchange.json')
    opps_data = get_data_file(opps_path)
    
    if not opps_data:
        # Fallback to mock data if real data isn't available
        opportunities = get_mock_opportunities('inter_exchange')
        return jsonify(opportunities)
    
    return jsonify(opps_data)

@api.route('/statistics')
def statistics():
    """Renvoie les statistiques du bot"""
    stats_path = os.path.join(current_app.config['DATA_DIR'], 'statistics', 'current_stats.json')
    stats_data = get_data_file(stats_path)
    
    if not stats_data:
        # Fallback to mock data if real data isn't available
        stats = get_mock_stats()
        return jsonify(stats)
    
    return jsonify(stats_data)

@api.route('/config', methods=['GET'])
def get_config():
    """Renvoie la configuration du bot"""
    config_path = os.path.join(os.getcwd(), 'config', 'arbitrage_config.json')
    
    if not os.path.exists(config_path):
        return jsonify({'error': 'Configuration file not found'}), 404
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Masquer les informations sensibles (tokens, mots de passe)
    if 'notifications' in config and 'telegram' in config['notifications']:
        if 'token' in config['notifications']['telegram']:
            config['notifications']['telegram']['token'] = '********'
    
    if 'notifications' in config and 'email' in config['notifications']:
        if 'password' in config['notifications']['email']:
            config['notifications']['email']['password'] = '********'
    
    return jsonify(config)

@api.route('/config', methods=['POST'])
def update_config():
    """
    Met à jour la configuration du bot
    
    Attend un objet JSON avec les modifications à appliquer
    """
    config_path = os.path.join(os.getcwd(), 'config', 'arbitrage_config.json')
    
    if not os.path.exists(config_path):
        return jsonify({'error': 'Configuration file not found'}), 404
    
    # Charger la configuration actuelle
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Appliquer les modifications
    updates = request.json
    if not updates:
        return jsonify({'error': 'No update data provided'}), 400
    
    # Mise à jour récursive de la configuration
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    # Créer une copie de sauvegarde
    backup_path = f"{config_path}.{int(time.time())}.bak"
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Appliquer les mises à jour
    config = update_dict(config, updates)
    
    # Sauvegarder la configuration mise à jour
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return jsonify({'success': True, 'message': 'Configuration updated'})

@api.route('/control/start', methods=['POST'])
def start_bot():
    """Démarre le bot"""
    try:
        # Dans une implémentation réelle, démarrer le bot
        # Exemple: subprocess.Popen(['python', 'run_arbitrage.py'])
        
        # Mettre à jour le statut
        status_path = os.path.join(current_app.config['DATA_DIR'], 'status', 'current_status.json')
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        
        status_data = get_data_file(status_path)
        if not status_data:
            status_data = {'status': {}}
        
        status_data['status']['active'] = True
        status_data['status']['start_time'] = datetime.now().isoformat()
        status_data['timestamp'] = datetime.now().isoformat()
        
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Bot started'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error starting bot: {str(e)}'}), 500

@api.route('/control/stop', methods=['POST'])
def stop_bot():
    """Arrête le bot"""
    try:
        # Dans une implémentation réelle, arrêter le bot
        # Exemple: Envoyer un signal au processus
        
        # Mettre à jour le statut
        status_path = os.path.join(current_app.config['DATA_DIR'], 'status', 'current_status.json')
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        
        status_data = get_data_file(status_path)
        if not status_data:
            status_data = {'status': {}}
        
        status_data['status']['active'] = False
        status_data['timestamp'] = datetime.now().isoformat()
        
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Bot stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error stopping bot: {str(e)}'}), 500

@api.route('/notifications/test', methods=['POST'])
def test_notification():
    """Envoie une notification de test"""
    channel = request.json.get('channel', 'log')
    
    try:
        # Dans une implémentation réelle, envoyer une notification de test
        # Importation du gestionnaire de notifications
        import sys
        import os
        sys.path.append(os.getcwd())
        
        try:
            from modules.notifications.notification_manager import NotificationManager
            from modules.notifications.notification_factory import NotificationFactory
            
            # Charger la configuration
            config_path = os.path.join(os.getcwd(), 'config', 'arbitrage_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Créer un gestionnaire de notifications
            factory = NotificationFactory()
            manager = NotificationManager(factory, config.get('notifications', {}))
            
            # Envoyer une notification de test
            if channel == 'all':
                manager.send_notification(
                    "Test de notification",
                    f"Ceci est un test de notification depuis l'interface web.\nHeure: {datetime.now().isoformat()}"
                )
            else:
                manager.send_notification_to_channel(
                    channel,
                    "Test de notification",
                    f"Ceci est un test de notification depuis l'interface web.\nHeure: {datetime.now().isoformat()}"
                )
            
            return jsonify({
                'success': True,
                'message': f'Test notification sent to {channel}'
            })
        
        except ImportError:
            return jsonify({
                'success': False,
                'message': f'Notification module not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error sending test notification: {str(e)}'
        }), 500
