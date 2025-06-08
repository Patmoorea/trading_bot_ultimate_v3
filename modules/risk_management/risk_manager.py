#!/usr/bin/env python3
"""
Module de gestion des risques pour les opérations d'arbitrage
Créé: 2025-05-23
@author: Patmoorea
"""
import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration du logger
logger = logging.getLogger(__name__)

class RiskManager:
    """
    Gestionnaire de risques pour les opérations d'arbitrage et de trading
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le gestionnaire de risques
        
        Args:
            config: Configuration du gestionnaire de risques
        """
        self.config = config or {}
        
        # Paramètres par défaut de gestion des risques
        self.max_position_size = self.config.get('max_position_size', 0.05)  # 5% du capital
        self.max_open_positions = self.config.get('max_open_positions', 3)  # Max 3 positions ouvertes
        self.max_daily_trades = self.config.get('max_daily_trades', 20)  # Max 20 trades par jour
        self.max_daily_drawdown = self.config.get('max_daily_drawdown', 0.02)  # 2% max de drawdown quotidien
        self.max_total_drawdown = self.config.get('max_total_drawdown', 0.10)  # 10% max de drawdown total
        self.position_sizing_method = self.config.get('position_sizing_method', 'fixed')  # Méthode de dimensionnement
        
        # État de la gestion des risques
        self.open_positions = {}
        self.daily_trades = {}
        self.daily_pnl = {}
        self.total_pnl = Decimal('0')
        self.initial_capital = Decimal(str(self.config.get('initial_capital', '10000')))
        self.current_capital = self.initial_capital
        self.high_watermark = self.initial_capital
        self.trade_history = []
        
        # Journalisation du démarrage
        logger.info("Gestionnaire de risques initialisé avec la configuration suivante:")
        logger.info(f"- Capital initial: {self.initial_capital}")
        logger.info(f"- Taille max de position: {self.max_position_size * 100}% du capital")
        logger.info(f"- Max positions ouvertes: {self.max_open_positions}")
        logger.info(f"- Max trades quotidiens: {self.max_daily_trades}")
        logger.info(f"- Max drawdown quotidien: {self.max_daily_drawdown * 100}%")
        logger.info(f"- Max drawdown total: {self.max_total_drawdown * 100}%")
    
    def check_arbitrage_risk(self, 
                           opportunity: Dict[str, Any], 
                           balance: Dict[str, Decimal]) -> Dict[str, Any]:
        """
        Évalue les risques d'une opportunité d'arbitrage
        
        Args:
            opportunity: Opportunité d'arbitrage à évaluer
            balance: Solde actuel des différentes devises
            
        Returns:
            Résultat de l'évaluation des risques
        """
        # Extraire les données de l'opportunité
        symbol = opportunity.get('symbol', '')
        spread = Decimal(str(opportunity.get('spread', 0)))
        volume = Decimal(str(opportunity.get('volume', 0)))
        timestamp = opportunity.get('timestamp', datetime.now())
        
        # Extraire la paire base/quote
        if '/' in symbol:
            base, quote = symbol.split('/')
        else:
            logger.warning(f"Format de symbole invalide: {symbol}")
            return {'approved': False, 'reason': 'Format de symbole invalide'}
        
        # Vérifier si nous avons assez de fonds
        if quote not in balance or balance[quote] <= 0:
            return {'approved': False, 'reason': f'Solde insuffisant en {quote}'}
        
        # Vérifier le nombre de positions ouvertes
        current_date = timestamp.date() if isinstance(timestamp, datetime) else datetime.now().date()
        if len(self.open_positions) >= self.max_open_positions:
            return {'approved': False, 'reason': 'Nombre maximum de positions ouvertes atteint'}
        
        # Vérifier le nombre de trades quotidiens
        day_key = current_date.isoformat()
        if day_key in self.daily_trades and self.daily_trades[day_key] >= self.max_daily_trades:
            return {'approved': False, 'reason': 'Nombre maximum de trades quotidiens atteint'}
        
        # Vérifier le drawdown quotidien
        if day_key in self.daily_pnl:
            daily_drawdown = -self.daily_pnl[day_key] / self.current_capital
            if daily_drawdown > self.max_daily_drawdown:
                return {'approved': False, 'reason': f'Drawdown quotidien maximum atteint ({daily_drawdown:.2%})'}
        
        # Vérifier le drawdown total
        total_drawdown = (self.high_watermark - self.current_capital) / self.high_watermark
        if total_drawdown > self.max_total_drawdown:
            return {'approved': False, 'reason': f'Drawdown total maximum atteint ({total_drawdown:.2%})'}
        
        # Calculer la taille de position recommandée
        position_size = self._calculate_position_size(quote, balance[quote], spread)
        
        # Ajuster le volume si nécessaire
        adjusted_volume = min(volume, position_size)
        if adjusted_volume <= 0:
            return {'approved': False, 'reason': 'Volume d\'arbitrage trop faible'}
        
        # Calculer le risque estimé (incluant les frais et le slippage)
        estimated_risk = self._estimate_arbitrage_risk(opportunity, adjusted_volume)
        
        # Approuver l'opportunité avec la taille ajustée
        return {
            'approved': True,
            'adjusted_volume': float(adjusted_volume),
            'original_volume': float(volume),
            'estimated_risk': float(estimated_risk),
            'max_loss': float(estimated_risk * adjusted_volume),
            'risk_reward_ratio': float(spread / estimated_risk) if estimated_risk > 0 else float('inf')
        }
    
    def _calculate_position_size(self, 
                              currency: str, 
                              available_balance: Decimal,
                              spread: Decimal) -> Decimal:
        """
        Calcule la taille de position optimale selon la méthode configurée
        
        Args:
            currency: Devise de la position
            available_balance: Solde disponible dans cette devise
            spread: Spread de l'opportunité (en décimal, pas en %)
            
        Returns:
            Taille de position recommandée
        """
        method = self.position_sizing_method.lower()
        
        # Méthode de base: pourcentage fixe du capital
        if method == 'fixed':
            return available_balance * Decimal(str(self.max_position_size))
        
        # Méthode de Kelly (simplifiée)
        elif method == 'kelly':
            # Estimer la probabilité de succès (peut être affiné)
            win_prob = Decimal('0.6')  # 60% de chances de succès par défaut
            
            # Formule de Kelly: f* = (p * b - q) / b
            # où p = proba de gain, q = proba de perte, b = ratio gains/pertes
            
            # Pour l'arbitrage, on peut considérer le spread comme le gain potentiel
            # et une valeur fixe comme perte potentielle (ex: 0.1%)
            potential_loss = Decimal('0.001')  # 0.1%
            
            if spread <= 0 or potential_loss <= 0:
                return Decimal('0')
                
            kelly_fraction = (win_prob * spread - (1 - win_prob)) / spread
            
            # Limiter la fraction de Kelly (demi-Kelly est souvent plus prudent)
            kelly_fraction = kelly_fraction * Decimal('0.5')
            
            # Limiter à notre maximum configuré
            kelly_fraction = min(kelly_fraction, Decimal(str(self.max_position_size)))
            
            # Ne pas accepter de fraction négative
            kelly_fraction = max(kelly_fraction, Decimal('0'))
            
            return available_balance * kelly_fraction
        
        # Méthode de risque fixe (% du capital risqué constant)
        elif method == 'fixed_risk':
            risk_per_trade = Decimal('0.005')  # 0.5% du capital risqué par trade
            estimated_max_loss = Decimal('0.001')  # 0.1% de perte maximale estimée
            
            if estimated_max_loss <= 0:
                return Decimal('0')
                
            position_size = (available_balance * risk_per_trade) / estimated_max_loss
            
            # Limiter à notre maximum configuré
            max_size = available_balance * Decimal(str(self.max_position_size))
            position_size = min(position_size, max_size)
            
            return position_size
        
        # Méthode par défaut
        else:
            return available_balance * Decimal(str(self.max_position_size))
    
    def _estimate_arbitrage_risk(self, opportunity: Dict[str, Any], volume: Decimal) -> Decimal:
        """
        Estime le risque maximum d'une opération d'arbitrage
        
        Args:
            opportunity: Opportunité d'arbitrage
            volume: Volume de l'opération
            
        Returns:
            Risque estimé (en pourcentage du capital)
        """
        # Paramètres de risque
        slippage_risk = Decimal('0.001')  # 0.1% de risque de slippage
        execution_risk = Decimal('0.002')  # 0.2% de risque d'exécution
        
        # Pour l'arbitrage, le risque principal est que le spread se réduise avant l'exécution
        # et que les frais ne soient pas couverts
        spread = Decimal(str(opportunity.get('spread', 0)))
        
        # Risque total estimé (pourcentage du capital)
        total_risk = slippage_risk + execution_risk + (spread * Decimal('0.2'))  # 20% du spread comme marge supplémentaire
        
        return total_risk
    
    def update_position(self, 
                      trade: Dict[str, Any]) -> None:
        """
        Met à jour l'état des positions après une opération
        
        Args:
            trade: Informations sur l'opération réalisée
        """
        timestamp = trade.get('timestamp', datetime.now())
        symbol = trade.get('symbol', '')
        trade_type = trade.get('type', '')
        volume = Decimal(str(trade.get('volume', 0)))
        price = Decimal(str(trade.get('price', 0)))
        pnl = Decimal(str(trade.get('pnl', 0)))
        
        current_date = timestamp.date() if isinstance(timestamp, datetime) else datetime.now().date()
        day_key = current_date.isoformat()
        
        # Mettre à jour le décompte des trades quotidiens
        if day_key not in self.daily_trades:
            self.daily_trades[day_key] = 1
        else:
            self.daily_trades[day_key] += 1
        
        # Mettre à jour le PnL quotidien
        if day_key not in self.daily_pnl:
            self.daily_pnl[day_key] = pnl
        else:
            self.daily_pnl[day_key] += pnl
        
        # Mettre à jour le capital et le high watermark
        self.current_capital += pnl
        if self.current_capital > self.high_watermark:
            self.high_watermark = self.current_capital
        
        # Mettre à jour l'historique des trades
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'type': trade_type,
            'volume': float(volume),
            'price': float(price),
            'pnl': float(pnl),
            'capital': float(self.current_capital)
        })
        
        # Mettre à jour les positions ouvertes
        if trade_type == 'open':
            position_id = f"{symbol}_{timestamp.timestamp()}" if isinstance(timestamp, datetime) else f"{symbol}_{time.time()}"
            self.open_positions[position_id] = {
                'symbol': symbol,
                'open_time': timestamp,
                'open_price': price,
                'volume': volume
            }
        elif trade_type == 'close' and 'position_id' in trade:
            position_id = trade['position_id']
            if position_id in self.open_positions:
                del self.open_positions[position_id]
    
    def update_arbitrage_trade(self, 
                             arbitrage_result: Dict[str, Any]) -> None:
        """
        Met à jour l'état après une opération d'arbitrage
        
        Args:
            arbitrage_result: Résultat de l'opération d'arbitrage
        """
        timestamp = arbitrage_result.get('timestamp', datetime.now())
        symbol = arbitrage_result.get('symbol', '')
        volume = Decimal(str(arbitrage_result.get('volume', 0)))
        profit = Decimal(str(arbitrage_result.get('profit', 0)))
        
        # Créer une entrée de trade pour l'opération d'arbitrage
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'type': 'arbitrage',
            'volume': volume,
            'price': 0,  # Pas pertinent pour l'arbitrage
            'pnl': profit
        }
        
        # Mettre à jour l'état avec ce trade
        self.update_position(trade)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Génère un rapport sur l'état actuel des risques
        
        Returns:
            Rapport détaillé sur l'état des risques
        """
        current_time = datetime.now()
        current_date = current_time.date()
        day_key = current_date.isoformat()
        
        # Calculer les métriques de risque actuelles
        open_position_count = len(self.open_positions)
        daily_trade_count = self.daily_trades.get(day_key, 0)
        daily_pnl_value = float(self.daily_pnl.get(day_key, Decimal('0')))
        daily_drawdown = -daily_pnl_value / float(self.current_capital) if daily_pnl_value < 0 else 0
        
        total_drawdown = float((self.high_watermark - self.current_capital) / self.high_watermark)
        
        # Calculer la capacité de trading restante
        remaining_positions = self.max_open_positions - open_position_count
        remaining_daily_trades = self.max_daily_trades - daily_trade_count
        remaining_daily_drawdown = self.max_daily_drawdown - daily_drawdown
        remaining_total_drawdown = self.max_total_drawdown - total_drawdown
        
        # Performance globale
        total_trades = sum(self.daily_trades.values())
        total_pnl = float(self.current_capital - self.initial_capital)
        roi = total_pnl / float(self.initial_capital)
        
        # Générer le rapport
        report = {
            'timestamp': current_time.isoformat(),
            'capital': {
                'initial': float(self.initial_capital),
                'current': float(self.current_capital),
                'high_watermark': float(self.high_watermark)
            },
            'positions': {
                'open_count': open_position_count,
                'max_allowed': self.max_open_positions,
                'remaining': remaining_positions
            },
            'daily_metrics': {
                'trade_count': daily_trade_count,
                'max_allowed': self.max_daily_trades,
                'remaining_trades': remaining_daily_trades,
                'pnl': daily_pnl_value,
                'drawdown': daily_drawdown,
                'max_drawdown_allowed': self.max_daily_drawdown,
                'remaining_drawdown': remaining_daily_drawdown
            },
            'total_metrics': {
                'trade_count': total_trades,
                'pnl': total_pnl,
                'roi': roi,
                'drawdown': total_drawdown,
                'max_drawdown_allowed': self.max_total_drawdown,
                'remaining_drawdown': remaining_total_drawdown
            },
            'risk_status': {
                'trade_limit_reached': daily_trade_count >= self.max_daily_trades,
                'position_limit_reached': open_position_count >= self.max_open_positions,
                'daily_drawdown_warning': daily_drawdown >= 0.8 * self.max_daily_drawdown,
                'total_drawdown_warning': total_drawdown >= 0.8 * self.max_total_drawdown
            }
        }
        
        return report
    
    def calculate_stop_loss(self, 
                          symbol: str, 
                          entry_price: Decimal, 
                          position_size: Decimal,
                          risk_per_trade: Optional[Decimal] = None) -> Dict[str, float]:
        """
        Calcule le niveau de stop loss recommandé pour une position
        
        Args:
            symbol: Symbole de la position
            entry_price: Prix d'entrée
            position_size: Taille de la position
            risk_per_trade: Risque maximum par trade (en % du capital)
            
        Returns:
            Niveaux de stop loss recommandés (absolu et en %)
        """
        # Risque par défaut: 1% du capital
        if risk_per_trade is None:
            risk_per_trade = Decimal('0.01')
        
        # Calculer le montant maximum à risquer
        max_risk_amount = self.current_capital * risk_per_trade
        
        # Calculer le stop loss (simplifié)
        if position_size <= 0 or entry_price <= 0:
            return {'stop_loss_price': 0, 'stop_loss_percent': 0}
        
        # Pourcentage maximal de mouvement de prix accepté
        max_price_move_percent = max_risk_amount / (position_size * entry_price)
        
        # Calculer le prix du stop loss
        stop_loss_price = entry_price * (Decimal('1') - max_price_move_percent)
        
        return {
            'stop_loss_price': float(stop_loss_price),
            'stop_loss_percent': float(max_price_move_percent * Decimal('100'))
        }
    
    def save_state(self, file_path: str) -> None:
        """
        Sauvegarde l'état actuel du gestionnaire de risques
        
        Args:
            file_path: Chemin où sauvegarder l'état
        """
        state = {
            'config': self.config,
            'current_capital': float(self.current_capital),
            'high_watermark': float(self.high_watermark),
            'open_positions': {k: {kk: float(vv) if isinstance(vv, Decimal) else vv 
                                  for kk, vv in v.items()} 
                              for k, v in self.open_positions.items()},
            'daily_trades': self.daily_trades,
            'daily_pnl': {k: float(v) for k, v in self.daily_pnl.items()},
            'trade_history': self.trade_history
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, default=self._json_serializer)
            logger.info(f"État du gestionnaire de risques sauvegardé dans {file_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état: {e}")
    
    def load_state(self, file_path: str) -> bool:
        """
        Charge l'état du gestionnaire de risques depuis un fichier
        
        Args:
            file_path: Chemin du fichier d'état
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restaurer l'état
            self.config = state.get('config', self.config)
            self.current_capital = Decimal(str(state.get('current_capital', self.initial_capital)))
            self.high_watermark = Decimal(str(state.get('high_watermark', self.initial_capital)))
            
            # Restaurer les positions ouvertes
            self.open_positions = {}
            for k, v in state.get('open_positions', {}).items():
                self.open_positions[k] = {kk: Decimal(str(vv)) if kk in ['open_price', 'volume'] else vv 
                                         for kk, vv in v.items()}
            
            # Restaurer les compteurs quotidiens
            self.daily_trades = state.get('daily_trades', {})
            self.daily_pnl = {k: Decimal(str(v)) for k, v in state.get('daily_pnl', {}).items()}
            
            # Restaurer l'historique des trades
            self.trade_history = state.get('trade_history', [])
            
            logger.info(f"État du gestionnaire de risques chargé depuis {file_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {e}")
            return False
    
    def _json_serializer(self, obj):
        """Fonction d'aide pour sérialiser des types non-JSON"""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type non sérialisable: {type(obj)}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configuration du gestionnaire de risques
    config = {
        'initial_capital': 10000,
        'max_position_size': 0.05,
        'max_open_positions': 3,
        'max_daily_trades': 20,
        'max_daily_drawdown': 0.02,
        'max_total_drawdown': 0.10,
        'position_sizing_method': 'kelly'
    }
    
    # Initialiser le gestionnaire de risques
    risk_manager = RiskManager(config)
    
    # Exemple d'opportunité d'arbitrage
    opportunity = {
        'symbol': 'BTC/USDT',
        'spread': 0.005,  # 0.5%
        'volume': 0.1,     # 0.1 BTC
        'timestamp': datetime.now()
    }
    
    # Solde disponible
    balance = {'USDT': Decimal('10000'), 'BTC': Decimal('0')}
    
    # Vérifier les risques de l'opportunité
    risk_assessment = risk_manager.check_arbitrage_risk(opportunity, balance)
    print(f"Évaluation des risques: {json.dumps(risk_assessment, indent=2)}")
    
    # Simuler un trade réussi
    if risk_assessment['approved']:
        # Simuler un profit de 0.5%
        profit = Decimal('0.005') * Decimal(str(risk_assessment['adjusted_volume'])) * Decimal('50000')  # 50000 = prix BTC
        
        # Mettre à jour l'état
        arbitrage_result = {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USDT',
            'volume': risk_assessment['adjusted_volume'],
            'profit': profit
        }
        
        risk_manager.update_arbitrage_trade(arbitrage_result)
        
        # Obtenir un rapport sur l'état des risques
        risk_report = risk_manager.get_risk_report()
        print(f"Rapport de risques: {json.dumps(risk_report, indent=2)}")
        
        # Sauvegarder l'état
        risk_manager.save_state('risk_manager_state.json')
