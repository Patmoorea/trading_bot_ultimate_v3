import os
import sys
import logging
import asyncio
from datetime import datetime
import numpy as np
import ccxt
from dotenv import load_dotenv

# Ajout des chemins pour les modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(current_dir)

# Imports de vos modules existants
from src.core.websocket.multi_stream import MultiStreamManager, StreamConfig
from src.core.buffer.circular_buffer import CircularBuffer
from src.indicators.advanced.multi_timeframe import MultiTimeframeAnalyzer, TimeframeConfig
from src.analysis.indicators.orderflow.orderflow_analysis import OrderFlowAnalysis
from src.analysis.indicators.volume.volume_analysis import VolumeAnalysis
from src.analysis.indicators.volatility.volatility import VolatilityIndicators
from src.ai.cnn_lstm import CNNLSTM
from src.ai_decision.ppo_transformer import PPOTradingAgent as PPOTransformer
from src.risk_management.circuit_breakers import CircuitBreaker
from src.risk_management.position_manager import PositionManager
from src.core.exchange import ExchangeInterface as Exchange
from src.news_processor.sentiment import NewsSentimentAnalyzer
from src.notifications.telegram_bot import TelegramBot
from src.regime_detection.hmm_kmeans import MarketRegimeDetector
from src.strategies.arbitrage.service import ArbitrageEngine
from src.monitoring.dashboard import TradingDashboard
from src.liquidity_heatmap.generator import HeatmapGenerator

# Configuration
load_dotenv()
config = {
    "TRADING": {
        "base_currency": "USDC",
        "pairs": ["BTC/USDC", "ETH/USDC"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "study_period": "7d"  # Période d'étude initiale
    },
    "RISK": {
        'max_drawdown': 0.05,        # 5% max
        'daily_stop_loss': 0.02,     # 2% par jour
        'position_sizing': 'volatility_based',  # Kelly modifié
        'circuit_breaker': {
            'market_crash': True,
            'liquidity_shock': True,
            'black_swan': True
        }
    },
    "AI": {
        "confidence_threshold": 0.75,
        "min_training_size": 1000
    }
}

# Initialisation du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBotM4:
    def __init__(self):
        # Initialisation des composants de base
        self.exchange = Exchange()
        # Configuration du websocket
        stream_config = StreamConfig(
            max_connections=12,
            reconnect_delay=1.0,
            buffer_size=10000
        )
        self.websocket = MultiStreamManager(config=stream_config)
        self.buffer = CircularBuffer()  # Correction de l'indentation ici
        self.circuit_breaker = CircuitBreaker()
        self.position_manager = PositionManager()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.telegram = TelegramBot()
        self.regime_detector = MarketRegimeDetector()
        self.arbitrage_engine = ArbitrageEngine()
        self.dashboard = TradingDashboard()
        self.heatmap = HeatmapGenerator()
        
        # Initialisation des modèles IA
        self.technical_model = CNNLSTMModel()
        self.decision_model = PPOTransformer()
        
        # Initialisation des indicateurs avancés
        self.timeframe_config = TimeframeConfig(
            timeframes=config["TRADING"]["timeframes"],
            weights={
                "1m": 0.1, "5m": 0.15, "15m": 0.2,
                "1h": 0.25, "4h": 0.15, "1d": 0.15
            }
        )
        self.advanced_indicators = MultiTimeframeAnalyzer(config=self.timeframe_config)
        self.orderflow_analysis = OrderFlowAnalysis()
        self.volume_analysis = VolumeAnalysis()
        self.volatility_indicators = VolatilityIndicators()
        
        # Dictionnaire des 42 indicateurs
        self.indicators = {
            "trend": {
                "supertrend": self._calculate_supertrend,
                "ichimoku": self._calculate_ichimoku,
                "vwma": self._calculate_vwma,
                "ema_ribbon": self._calculate_ema_ribbon,
                "parabolic_sar": self._calculate_psar,
                "zigzag": self._calculate_zigzag
            },
            "momentum": {
                "rsi": self._calculate_rsi,
                "stoch_rsi": self._calculate_stoch_rsi,
                "macd": self._calculate_macd,
                "awesome": self._calculate_ao,
                "momentum": self._calculate_momentum,
                "tsi": self._calculate_tsi
            },
            "volatility": {
                "bbands": self._calculate_bbands,
                "keltner": self._calculate_keltner,
                "atr": self._calculate_atr,
                "vix_fix": self._calculate_vix_fix,
                "natr": self._calculate_natr,
                "true_range": self._calculate_tr
            },
            "volume": {
                "obv": self._calculate_obv,
                "vwap": self._calculate_vwap,
                "acc_dist": self._calculate_ad,
                "chaikin_money": self._calculate_cmf,
                "ease_move": self._calculate_emv,
                "volume_profile": self._calculate_vp
            },
            "orderflow": {
                "delta": self._calculate_delta,
                "cvd": self._calculate_cvd,
                "footprint": self._calculate_footprint,
                "liquidity": self._calculate_liquidity,
                "imbalance": self._calculate_imbalance,
                "absorption": self._calculate_absorption
            }
        }

    async def initialize(self):
        """Initialisation des composants"""
        logger.info("🚀 Démarrage du Trading Bot M4...")
        await self.websocket.connect()
        await self.telegram.send_message("🤖 Bot démarré et prêt à trader!")

    async def study_market(self, period="7d"):
        """Analyse initiale du marché"""
        logger.info("📊 Étude du marché en cours...")
        historical_data = await self.exchange.get_historical_data(
            config["TRADING"]["pairs"],
            config["TRADING"]["timeframes"],
            period
        )
        
        # Calcul des indicateurs sur l'historique
        indicators_analysis = {}
        for timeframe in config["TRADING"]["timeframes"]:
            tf_data = historical_data[timeframe]
            indicators_analysis[timeframe] = self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
        
        # Détection du régime de marché
        regime = self.regime_detector.detect_regime(indicators_analysis)
        logger.info(f"📈 Régime de marché détecté: {regime}")
        
        # Envoi du rapport complet sur Telegram
        analysis_report = self._generate_analysis_report(indicators_analysis, regime)
        await self.telegram.send_message(analysis_report)
        
        return regime, historical_data, indicators_analysis

    def _generate_analysis_report(self, indicators_analysis, regime):
        """Génère un rapport d'analyse détaillé"""
        report = [
            "📊 Analyse complète du marché:",
            f"Régime: {regime}",
            "\nTendances principales:"
        ]
        
        for timeframe, analysis in indicators_analysis.items():
            report.append(f"\n{timeframe}:")
            trend_strength = analysis['trend'].get('trend_strength', 0)
            volatility = analysis['volatility'].get('current_volatility', 0)
            volume_profile = analysis['volume'].get('volume_profile', {})
            
            report.extend([
                f"- Force de la tendance: {trend_strength:.2%}",
                f"- Volatilité: {volatility:.2%}",
                f"- Volume: {volume_profile.get('strength', 'N/A')}"
            ])
        
        return "\n".join(report)

    async def process_market_data(self):
        """Traitement des données de marché avec tous les indicateurs"""
        try:
            # Récupération des données
            market_data = self.buffer.get_latest()
            
            # Calcul de tous les indicateurs
            indicators_results = {}
            for timeframe in config["TRADING"]["timeframes"]:
                tf_data = market_data[timeframe]
                indicators_results[timeframe] = self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
            
            # Génération de la heatmap
            heatmap = self.heatmap.generate_heatmap(
                await self.exchange.get_orderbook(config["TRADING"]["pairs"])
            )
            
            # Notification des signaux importants
            await self._notify_significant_signals(indicators_results)
            
            # Mise à jour du dashboard
            self.dashboard.update(market_data, indicators_results, heatmap)
            
            return market_data, indicators_results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {e}")
            await self.telegram.send_message(f"⚠️ Erreur traitement: {str(e)}")
            return None, None

    async def _notify_significant_signals(self, indicators_results):
        """Notifie les signaux importants sur Telegram"""
        for timeframe, analysis in indicators_results.items():
            for category, indicators in analysis.items():
                for indicator, value in indicators.items():
                    if self._is_significant_signal(category, indicator, value):
                        await self.telegram.send_message(
                            f"⚠️ Signal important détecté!\n"
                            f"Timeframe: {timeframe}\n"
                            f"Catégorie: {category}\n"
                            f"Indicateur: {indicator}\n"
                            f"Valeur: {value}"
                        )

    def _is_significant_signal(self, category, indicator, value):
        """Détermine si un signal est significatif"""
        thresholds = {
            'trend': {
                'supertrend': 0.8,
                'ichimoku': 0.7
            },
            'momentum': {
                'rsi': {'oversold': 30, 'overbought': 70},
                'macd': 0
            },
            'volatility': {
                'bbands': 2.0,
                'atr': 0.02
            }
        }
        
        if category in thresholds and indicator in thresholds[category]:
            threshold = thresholds[category][indicator]
            if isinstance(threshold, dict):
                return value < threshold['oversold'] or value > threshold['overbought']
            return abs(value) > threshold
            
        return False

    async def analyze_signals(self, market_data, indicators):
        """Analyse technique et fondamentale avancée"""
        try:
            # Analyse technique via CNN-LSTM
            technical_score = self.technical_model.predict({
                'market_data': market_data,
                'indicators': indicators
            })
            
            # Analyse des news
            news_impact = await self.news_analyzer.analyze_recent_news()
            
            # Analyse multi-timeframe
            timeframe_analysis = self._analyze_multi_timeframe(indicators)
            
            # Détection du régime actuel
            current_regime = self.regime_detector.detect_regime(indicators)
            
            # Décision finale via PPO+Transformer
            decision = self.decision_model.make_decision(
                technical_score=technical_score,
                news_sentiment=news_impact,
                market_regime=current_regime,
                timeframe_analysis=timeframe_analysis
            )
            
            # Ajout des stop loss et take profit
            decision = self._add_risk_management(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Erreur analyse signaux: {e}")
            await self.telegram.send_message(f"⚠️ Erreur analyse: {str(e)}")
            return None

    def _analyze_multi_timeframe(self, indicators):
        """Analyse multi-timeframe des indicateurs"""
        analysis = {}
        for timeframe in config["TRADING"]["timeframes"]:
            weight = self.timeframe_config.weights[timeframe]
            if timeframe in indicators:
                tf_indicators = indicators[timeframe]
                analysis[timeframe] = {
                    'weight': weight,
                    'trend_score': self._calculate_trend_score(tf_indicators),
                    'momentum_score': self._calculate_momentum_score(tf_indicators),
                    'volatility_score': self._calculate_volatility_score(tf_indicators),
                    'volume_score': self._calculate_volume_score(tf_indicators)
                }
        return analysis

    def _add_risk_management(self, decision):
        """Ajoute les paramètres de gestion des risques"""
        if decision and decision.get('action') == 'buy':
            entry_price = decision['entry_price']
            decision.update({
                'stop_loss': entry_price * (1 - config["RISK"]['daily_stop_loss']),
                'take_profit': entry_price * (1 + config["RISK"]['daily_stop_loss'] * 2),
                'trailing_stop': {
                    'activation_price': entry_price * 1.01,
                    'callback_rate': 0.007
                }
            })
        return decision

    async def execute_trades(self, decision):
        """Exécution des trades selon la décision"""
        if await self.circuit_breaker.should_stop_trading():
            logger.warning("🛑 Circuit breaker activé - Trading suspendu")
            await self.telegram.send_message("⚠️ Trading suspendu: Circuit breaker activé")
            return
        
        if decision and decision["confidence"] > config["AI"]["confidence_threshold"]:
            try:
                # Vérification des opportunités d'arbitrage
                arb_ops = await self.arbitrage_engine.find_opportunities()
                if arb_ops:
                    await self.telegram.send_message(
                        f"💰 Opportunité d'arbitrage détectée:\n{arb_ops}"
                    )
                
                # Calcul de la taille de position
                position_size = self.position_manager.calculate_position_size(
                    decision,
                    available_balance=await self.exchange.get_balance("USDC")
                )
                
                # Placement de l'ordre avec stop loss
                order = await self.exchange.create_order(
                    symbol=decision["symbol"],
                    type="limit",
                    side="buy",  # Achat uniquement comme demandé
                    amount=position_size,
                    price=decision["entry_price"],
                    params={
                        "stopLoss": {
                            "type": "trailing",
                            "activation_price": decision["trailing_stop"]["activation_price"],
                            "callback_rate": decision["trailing_stop"]["callback_rate"]
                        }
                    }
                )
                
                # Notification Telegram détaillée
                await self.telegram.send_message(
                    f"🔄 Ordre placé:\n"
                    f"Symbol: {order['symbol']}\n"
                    f"Type: {order['type']}\n"
                    f"Prix: {order['price']}\n"
                    f"Stop Loss: {decision['stop_loss']}\n"
                    f"Take Profit: {decision['take_profit']}\n"
                    f"Trailing Stop: {decision['trailing_stop']['activation_price']}\n"
                    f"Confiance: {decision['confidence']:.2%}\n"
                    f"Volume: {position_size} {config['TRADING']['base_currency']}"
                )
                
            except Exception as e:
                logger.error(f"Erreur lors de l'exécution: {e}")
                await self.telegram.send_message(f"⚠️ Erreur d'exécution: {str(e)}")

    async def run(self):
        """Boucle principale du bot"""
        await self.initialize()
        
        # Étude initiale du marché
        regime, historical_data, initial_analysis = await self.study_market(
            config["TRADING"]["study_period"]
        )
        
        while True:
            try:
                # 1. Traitement des données
                market_data, indicators = await self.process_market_data()
                
                # 2. Analyse et décision
                decision = await self.analyze_signals(market_data, indicators)
                
                # 3. Exécution si nécessaire
                if decision:
                    await self.execute_trades(decision)
                
                # Attente avant la prochaine itération
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("👋 Arrêt manuel demandé")
                await self.telegram.send_message("🛑 Bot arrêté manuellement")
                break
            except Exception as e:
                logger.error(f"Erreur critique: {e}")
                await self.telegram.send_message(f"🚨 Erreur critique: {str(e)}")
                await asyncio.sleep(5)

async def main():
    bot = TradingBotM4()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
