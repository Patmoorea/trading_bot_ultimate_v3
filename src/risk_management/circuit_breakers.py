import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self):
        self.triggers = {
            'market_crash': {
                'threshold': -0.1,  # -10% de chute
                'timeframe': timedelta(minutes=5),
                'triggered': False,
                'last_trigger': None,
                'cooldown': timedelta(hours=1)
            },
            'liquidity_shock': {
                'threshold': 0.5,  # 50% de baisse de liquidité
                'timeframe': timedelta(minutes=15),
                'triggered': False,
                'last_trigger': None,
                'cooldown': timedelta(minutes=30)
            },
            'black_swan': {
                'threshold': -0.2,  # -20% de chute
                'timeframe': timedelta(minutes=1),
                'triggered': False,
                'last_trigger': None,
                'cooldown': timedelta(hours=24)
            }
        }
        self.is_active = False
        self.last_check = datetime.utcnow()
        
    async def should_stop_trading(self) -> bool:
        """
        Vérifie si le trading doit être arrêté en fonction des conditions du marché
        """
        try:
            for trigger_name, trigger in self.triggers.items():
                if trigger['triggered']:
                    if datetime.utcnow() - trigger['last_trigger'] < trigger['cooldown']:
                        logger.warning(f"Circuit breaker {trigger_name} actif")
                        return True
                    else:
                        trigger['triggered'] = False
                        
            # Vérification des conditions de marché
            market_conditions = await self._check_market_conditions()
            
            for condition_name, condition in market_conditions.items():
                trigger = self.triggers.get(condition_name)
                if trigger and condition <= trigger['threshold']:
                    trigger['triggered'] = True
                    trigger['last_trigger'] = datetime.utcnow()
                    logger.warning(f"Circuit breaker activé: {condition_name}")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Erreur circuit breaker: {str(e)}")
            return True  # Par sécurité, on arrête le trading en cas d'erreur
            
    async def _check_market_conditions(self) -> Dict[str, float]:
        """
        Vérifie les conditions actuelles du marché
        Retourne un dictionnaire avec les différentes métriques
        """
        # Simulé pour l'exemple - À implémenter avec vos données réelles
        return {
            'market_crash': 0.0,  # Pas de crash
            'liquidity_shock': 1.0,  # Liquidité normale
            'black_swan': 0.0  # Pas d'événement extrême
        }
        
    def reset(self):
        """
        Réinitialise tous les circuit breakers
        """
        for trigger in self.triggers.values():
            trigger['triggered'] = False
            trigger['last_trigger'] = None
            
    def get_status(self) -> Dict[str, Dict]:
        """
        Retourne l'état actuel des circuit breakers
        """
        return {
            name: {
                'triggered': trigger['triggered'],
                'last_trigger': trigger['last_trigger'],
                'cooldown_remaining': (
                    trigger['cooldown'] - (datetime.utcnow() - trigger['last_trigger'])
                    if trigger['last_trigger']
                    else None
                )
            }
            for name, trigger in self.triggers.items()
        }
