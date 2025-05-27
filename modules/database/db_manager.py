"""
Module de gestion de la base de données SQLite
Pour stocker et récupérer les données d'opportunités d'arbitrage
Created: 2025-05-23 04:10:00
@author: Patmooreaoui
"""

import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

class DatabaseManager:
    """Gestionnaire de base de données SQLite pour le bot de trading"""
    
    def __init__(self, db_path: str = "data/trading_bot.db"):
        """
        Initialise le gestionnaire de base de données
        
        Args:
            db_path: Chemin vers le fichier de base de données SQLite
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Créer le dossier de données si nécessaire
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialiser la base de données
        self._init_database()
    
    def _init_database(self):
        """Initialise la base de données avec les tables nécessaires"""
        try:
            # Charger le schéma SQL
            schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
            with open(schema_path, "r") as f:
                schema_sql = f.read()
            
            # Exécuter le schéma
            conn = self._get_connection()
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()
            
            self.logger.info("Base de données initialisée avec succès")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
    
    def _get_connection(self):
        """Établit et retourne une connexion à la base de données"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par leur nom
        return conn
    
    def store_opportunity(self, opportunity: Dict):
        """
        Enregistre une opportunité d'arbitrage dans la base de données
        
        Args:
            opportunity: Dictionnaire décrivant l'opportunité d'arbitrage
        
        Returns:
            L'ID de l'opportunité enregistrée
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Préparer les données
            arb_type = opportunity.get('type', 'unknown')
            detected_at = datetime.utcnow().isoformat()
            profit = opportunity.get('profit', 0.0)
            
            # Données spécifiques selon le type d'arbitrage
            if arb_type == 'triangular':
                exchange_name = opportunity.get('exchange', '')
                details_json = json.dumps({
                    'path': opportunity.get('path', []),
                    'volumes': opportunity.get('volumes', {}),
                    'rates': opportunity.get('rates', {}),
                    'execution_time': opportunity.get('execution_time', 0.0)
                })
                buy_exchange = exchange_name
                sell_exchange = exchange_name
                symbol = opportunity.get('path', [''])[0] if opportunity.get('path') else ''
            
            elif arb_type == 'inter_exchange':
                buy_exchange = opportunity.get('buy_exchange', '')
                sell_exchange = opportunity.get('sell_exchange', '')
                symbol = opportunity.get('symbol', '')
                details_json = json.dumps({
                    'buy_price': opportunity.get('buy_price', 0.0),
                    'sell_price': opportunity.get('sell_price', 0.0),
                    'buy_volume': opportunity.get('buy_volume', 0.0),
                    'sell_volume': opportunity.get('sell_volume', 0.0),
                    'transfer_time': opportunity.get('transfer_time', 0.0),
                    'total_fees': opportunity.get('total_fees', 0.0)
                })
            else:
                details_json = json.dumps(opportunity)
                buy_exchange = opportunity.get('buy_exchange', '')
                sell_exchange = opportunity.get('sell_exchange', '')
                symbol = opportunity.get('symbol', '')
            
            # Insérer l'opportunité
            cursor.execute(
                """
                INSERT INTO opportunities (
                    type, detected_at, profit, buy_exchange, sell_exchange, 
                    symbol, details_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (arb_type, detected_at, profit, buy_exchange, sell_exchange, 
                symbol, details_json, 'detected')
            )
            
            opportunity_id = cursor.lastrowid
            conn.commit()
            
            self.logger.debug(f"Opportunité {arb_type} #{opportunity_id} enregistrée avec profit {profit}%")
            
            conn.close()
            return opportunity_id
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de l'opportunité: {e}")
            return None
    
    def update_opportunity_status(self, opportunity_id: int, status: str, 
                                execution_result: Optional[Dict] = None):
        """
        Met à jour le statut d'une opportunité et les résultats d'exécution
        
        Args:
            opportunity_id: ID de l'opportunité
            status: Nouveau statut ('detected', 'analyzing', 'executing', 'executed', 'failed', 'ignored')
            execution_result: Résultats de l'exécution le cas échéant
        
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Mettre à jour le statut
            if execution_result:
                result_json = json.dumps(execution_result)
                cursor.execute(
                    "UPDATE opportunities SET status = ?, execution_result = ?, updated_at = ? WHERE id = ?",
                    (status, result_json, datetime.utcnow().isoformat(), opportunity_id)
                )
            else:
                cursor.execute(
                    "UPDATE opportunities SET status = ?, updated_at = ? WHERE id = ?",
                    (status, datetime.utcnow().isoformat(), opportunity_id)
                )
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour du statut de l'opportunité #{opportunity_id}: {e}")
            return False
    
    def get_opportunities(self, limit: int = 100, offset: int = 0, 
                        arb_type: Optional[str] = None, 
                        min_profit: Optional[float] = None,
                        status: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        exchange: Optional[str] = None) -> List[Dict]:
        """
        Récupère les opportunités selon des critères de filtrage
        
        Args:
            limit: Nombre maximum d'opportunités à récupérer
            offset: Décalage pour la pagination
            arb_type: Type d'arbitrage ('triangular', 'inter_exchange')
            min_profit: Profit minimum
            status: Statut des opportunités
            start_date: Date de début (format ISO)
            end_date: Date de fin (format ISO)
            exchange: Nom de l'exchange concerné
        
        Returns:
            Liste des opportunités correspondant aux critères
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Construire la requête
            query = "SELECT * FROM opportunities WHERE 1=1"
            params = []
            
            if arb_type:
                query += " AND type = ?"
                params.append(arb_type)
            
            if min_profit is not None:
                query += " AND profit >= ?"
                params.append(min_profit)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND detected_at >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND detected_at <= ?"
                params.append(end_date)
            
            if exchange:
                query += " AND (buy_exchange = ? OR sell_exchange = ?)"
                params.extend([exchange, exchange])
            
            # Ajouter l'ordre et la limite
            query += " ORDER BY detected_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Exécuter la requête
            cursor.execute(query, params)
            
            # Construire les résultats
            opportunities = []
            for row in cursor.fetchall():
                opp = dict(row)
                
                # Convertir les détails JSON en dictionnaire
                if 'details_json' in opp and opp['details_json']:
                    opp['details'] = json.loads(opp['details_json'])
                    del opp['details_json']
                
                # Convertir les résultats d'exécution JSON en dictionnaire
                if 'execution_result' in opp and opp['execution_result']:
                    opp['execution_result'] = json.loads(opp['execution_result'])
                
                opportunities.append(opp)
            
            conn.close()
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des opportunités: {e}")
            return []
    
    def get_opportunity_by_id(self, opportunity_id: int) -> Optional[Dict]:
        """
        Récupère une opportunité par son ID
        
        Args:
            opportunity_id: ID de l'opportunité
        
        Returns:
            Dictionnaire de l'opportunité ou None si non trouvée
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM opportunities WHERE id = ?", (opportunity_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            opportunity = dict(row)
            
            # Convertir les détails JSON en dictionnaire
            if 'details_json' in opportunity and opportunity['details_json']:
                opportunity['details'] = json.loads(opportunity['details_json'])
                del opportunity['details_json']
            
            # Convertir les résultats d'exécution JSON en dictionnaire
            if 'execution_result' in opportunity and opportunity['execution_result']:
                opportunity['execution_result'] = json.loads(opportunity['execution_result'])
            
            conn.close()
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de l'opportunité #{opportunity_id}: {e}")
            return None
    
    def get_daily_stats(self, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les statistiques quotidiennes des opportunités
        
        Args:
            start_date: Date de début (format ISO)
            end_date: Date de fin (format ISO)
        
        Returns:
            Dictionnaire des statistiques
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Construire la requête de base
            query_params = []
            date_filter = ""
            
            if start_date:
                date_filter += " AND detected_at >= ?"
                query_params.append(start_date)
            
            if end_date:
                date_filter += " AND detected_at <= ?"
                query_params.append(end_date)
            
            # Récupérer le nombre total d'opportunités par type
            cursor.execute(
                f"""
                SELECT type, COUNT(*) as count, AVG(profit) as avg_profit, 
                       MAX(profit) as max_profit, SUM(profit) as total_profit
                FROM opportunities
                WHERE 1=1 {date_filter}
                GROUP BY type
                """,
                query_params
            )
            
            stats_by_type = {}
            for row in cursor.fetchall():
                stats_by_type[row['type']] = {
                    'count': row['count'],
                    'avg_profit': row['avg_profit'],
                    'max_profit': row['max_profit'],
                    'total_profit': row['total_profit']
                }
            
            # Récupérer les opportunités par statut
            cursor.execute(
                f"""
                SELECT status, COUNT(*) as count
                FROM opportunities
                WHERE 1=1 {date_filter}
                GROUP BY status
                """,
                query_params
            )
            
            stats_by_status = {}
            for row in cursor.fetchall():
                stats_by_status[row['status']] = row['count']
            
            # Récupérer les statistiques quotidiennes
            cursor.execute(
                f"""
                SELECT 
                    DATE(detected_at) as date,
                    COUNT(*) as count,
                    AVG(profit) as avg_profit,
                    MAX(profit) as max_profit,
                    SUM(profit) as total_profit
                FROM opportunities
                WHERE 1=1 {date_filter}
                GROUP BY DATE(detected_at)
                ORDER BY date DESC
                """,
                query_params
            )
            
            daily_stats = []
            for row in cursor.fetchall():
                daily_stats.append({
                    'date': row['date'],
                    'count': row['count'],
                    'avg_profit': row['avg_profit'],
                    'max_profit': row['max_profit'],
                    'total_profit': row['total_profit']
                })
            
            conn.close()
            
            # Construire le résultat final
            return {
                'by_type': stats_by_type,
                'by_status': stats_by_status,
                'daily': daily_stats,
                'total_count': sum(t['count'] for t in stats_by_type.values()) if stats_by_type else 0,
                'period_start': start_date,
                'period_end': end_date
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des statistiques: {e}")
            return {
                'by_type': {},
                'by_status': {},
                'daily': [],
                'total_count': 0,
                'error': str(e)
            }
    
    def clear_old_opportunities(self, days: int = 30) -> int:
        """
        Supprime les opportunités plus anciennes qu'un certain nombre de jours
        
        Args:
            days: Nombre de jours à conserver
        
        Returns:
            Nombre d'opportunités supprimées
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                DELETE FROM opportunities 
                WHERE detected_at < datetime('now', '-' || ? || ' days')
                """,
                (days,)
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Suppression de {deleted_count} opportunités anciennes")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression des opportunités anciennes: {e}")
            return 0
