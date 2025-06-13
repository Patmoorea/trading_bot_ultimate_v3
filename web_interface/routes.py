"""
Routes principales pour l'interface web du bot de trading
Created: 2025-05-23 04:25:00
@author: Patmoorea
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
import os
import json
from datetime import datetime

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Page d'accueil du dashboard"""
    return render_template('index.html', title='Dashboard')

@main.route('/arbitrage/triangular')
def triangular_arbitrage():
    """Page de gestion de l'arbitrage triangulaire"""
    return render_template('triangular_arbitrage.html', title='Arbitrage Triangulaire')

@main.route('/arbitrage/inter-exchange')
def inter_exchange_arbitrage():
    """Page de gestion de l'arbitrage inter-exchanges"""
    return render_template('inter_exchange_arbitrage.html', title='Arbitrage Inter-Exchanges')

@main.route('/notifications')
def notifications():
    """Page de gestion des notifications"""
    return render_template('notifications.html', title='Notifications')

@main.route('/settings')
def settings():
    """Page de configuration du bot"""
    return render_template('settings.html', title='Configuration')

@main.route('/logs')
def logs():
    """Page de consultation des logs"""
    log_files = []
    log_dir = os.path.join(os.getcwd(), 'logs')
    
    if os.path.exists(log_dir):
        for file in os.listdir(log_dir):
            if file.endswith('.log'):
                file_path = os.path.join(log_dir, file)
                modified_time = os.path.getmtime(file_path)
                size = os.path.getsize(file_path)
                log_files.append({
                    'name': file,
                    'path': file_path,
                    'modified': datetime.fromtimestamp(modified_time),
                    'size': size
                })
    
    return render_template('logs.html', title='Logs', log_files=log_files)

@main.route('/logs/<filename>')
def view_log(filename):
    """
    Affiche le contenu d'un fichier log
    
    Args:
        filename: Nom du fichier log à afficher
    """
    log_path = os.path.join(os.getcwd(), 'logs', filename)
    
    if not os.path.exists(log_path) or not filename.endswith('.log'):
        flash('Fichier log non trouvé', 'danger')
        return redirect(url_for('main.logs'))
    
    with open(log_path, 'r') as f:
        content = f.readlines()
    
    return render_template('view_log.html', title=f'Log - {filename}', 
                          filename=filename, content=content)
