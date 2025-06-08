import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from .technical_analysis import TechnicalAnalysis
from .ai_engine import AIDecisionEngine
from .risk_manager import RiskManager
from .order_execution import SmartOrderExecutor
from .portfolio_monitor import PortfolioMonitor
from .news_analyzer import NewsAnalyzer
from .telegram_integration import TelegramBot
from ..config import Config

class TradingOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ta = TechnicalAnalysis()
        self.ai = AIDecisionEngine()
        self.risk = RiskManager()
        self.executor = SmartOrderExecutor()
        self.portfolio = PortfolioMonitor()
        self.news = NewsAnalyzer()
        self.telegram = TelegramBot()
        self.start_time = datetime.now(timezone.utc)
        self.current_user = "Patmoorea"
        self.state = {
            'is_trading': False,
            'last_analysis': None,
            'last_trade': None,
            'market_state': 'analyzing'
        }

    async def start(self):
        """Démarre tous les services du trading bot"""
        try:
            self.logger.info("Starting Trading Bot Ultimate...")
            
            # Démarrer le bot Telegram
            await self.telegram.start()
            
            # Initialiser l'état du système
            await self._initialize_system()
            
            # Démarrer les tâches principales
            await asyncio.gather(
                self._market_analysis_loop(),
                self._news_monitoring_loop(),
                self._portfolio_update_loop(),
                self._system_health_check_loop()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {e}")
            await self.stop()

    async def _initialize_system(self):
        """Initialise le système avec les configurations nécessaires"""
        init_message = f"""
🚀 *Trading Bot Ultimate Started*

Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
User: {self.current_user}
Mode: USDC Buy-Only
Initial Setup: Complete

_System is now analyzing market conditions..._
"""
        await self.telegram.send_alert(init_message)

    async def _market_analysis_loop(self):
        """Boucle principale d'analyse du marché"""
        while True:
            try:
                # 1. Analyse technique
                technical_data = await self._get_market_data()
                ta_results = self.ta.calculate_all_indicators(technical_data)
                
                # 2. Analyse AI
                ai_decision = await self.ai.analyze_market({
                    'technical': ta_results,
                    'market_data': technical_data
                })
                
                # 3. Vérification des risques
                risk_assessment = await self.risk.check_risk_levels(
                    technical_data,
                    self.portfolio._get_portfolio_summary()
                )
                
                # 4. Décision de trading
                if self._should_trade(ai_decision, risk_assessment):
                    await self._execute_trade_decision(ai_decision, risk_assessment)
                
                # 5. Mise à jour du dashboard
                await self._update_dashboard({
                    'technical': ta_results,
                    'ai': ai_decision,
                    'risk': risk_assessment,
                    'portfolio': self.portfolio._get_portfolio_summary()
                })
                
                await asyncio.sleep(Config.ANALYSIS_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in market analysis loop: {e}")
                await asyncio.sleep(10)

    async def _news_monitoring_loop(self):
        """Surveille et analyse les nouvelles"""
        while True:
            try:
                news_analysis = await self.news.analyze_news()
                
                if news_analysis['status'] == 'success':
                    # Filtrer les nouvelles importantes
                    for news in news_analysis['important_news']:
                        if news['impact_score'] > Config.NEWS_IMPACT_THRESHOLD:
                            await self.telegram.send_news_alert(news)
                    
                    # Mise à jour du dashboard
                    await self._update_dashboard_news(news_analysis)
                
                await asyncio.sleep(Config.NEWS_UPDATE_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in news monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _portfolio_update_loop(self):
        """Met à jour le portfolio et vérifie les alertes"""
        while True:
            try:
                # Mise à jour du portfolio
                portfolio_summary = await self.portfolio._update_metrics()
                
                # Vérification des alertes
                if len(self.portfolio.portfolio['alerts']) > 0:
                    latest_alert = self.portfolio.portfolio['alerts'][-1]
                    await self.telegram.send_alert(latest_alert)
                
                # Mise à jour du dashboard
                await self._update_dashboard_portfolio(portfolio_summary)
                
                await asyncio.sleep(Config.PORTFOLIO_UPDATE_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(30)

    async def _system_health_check_loop(self):
        """Vérifie la santé du système"""
        while True:
            try:
                health_status = self._check_system_health()
                
                if not health_status['healthy']:
                    await self.telegram.send_alert(
                        f"⚠️ System Health Alert: {health_status['message']}",
                        importance="high"
                    )
                
                # Mise à jour du dashboard
                await self._update_dashboard_health(health_status)
                
                await asyncio.sleep(Config.HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)

    def _should_trade(self, ai_decision: Dict, risk_assessment: Dict) -> bool:
        """Détermine si une trade doit être exécuté"""
        return (
            ai_decision['action'] == 'BUY' and
            ai_decision['combined_confidence'] > Config.MIN_CONFIDENCE and
            risk_assessment['allow_trading'] and
            self.state['is_trading']
        )

    async def _execute_trade_decision(self, ai_decision: Dict, risk_assessment: Dict):
        """Exécute une décision de trading"""
        try:
            # Création de l'ordre
            order = {
                'symbol': ai_decision['symbol'],
                'side': 'BUY',
                'amount': risk_assessment['position_size'],
                'type': 'LIMIT',
                'price': ai_decision['target_price']
            }
            
            # Exécution de l'ordre
            execution_result = await self.executor.execute_order(
                order['symbol'],
                order['side'],
                order['amount'],
                self._get_orderbook(),
                self._get_market_data()
            )
            
            if execution_result['status'] == 'completed':
                # Mise à jour du portfolio
                await self.portfolio.update_portfolio({
                    **order,
                    **execution_result
                })
                
                # Notification Telegram
                await self.telegram.send_trade_alert({
                    **order,
                    'confidence': ai_decision['combined_confidence'],
                    'reason': ai_decision['reason'],
                    'risk_level': risk_assessment['risk_metrics']['risk_level']
                })
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            await self.telegram.send_alert(
                f"🚨 Trade Execution Error: {str(e)}",
                importance="high"
            )

    async def _update_dashboard(self, data: Dict):
        """Met à jour le dashboard en temps réel"""
        try:
            # Utiliser SocketIO pour envoyer les mises à jour au frontend
            from .. import socketio
            
            socketio.emit('dashboard_update', {
                'type': 'full_update',
                'data': {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'user': self.current_user,
                    **data
                }
            })
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard: {e}")

    def _check_system_health(self) -> Dict:
        """Vérifie l'état de santé du système"""
        import psutil
        
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            return {
                'healthy': cpu_usage < 80 and memory_usage < 80,
                'metrics': {
                    'cpu': cpu_usage,
                    'memory': memory_usage,
                    'uptime': (datetime.now(timezone.utc) - self.start_time).total_seconds()
                },
                'message': 'System operating normally'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f"Health check failed: {str(e)}"
            }

