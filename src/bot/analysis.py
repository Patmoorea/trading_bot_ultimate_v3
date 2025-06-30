import logging
from typing import Dict, Any
import pandas as pd


class RegimeDetector:
    """
    Détecteur de régimes de marché basé sur l'analyse d'indicateurs techniques.
    Détecte les cycles haussiers, baissiers ou de range.
    """

    def __init__(self):
        self.current_regime = None
        self.logger = logging.getLogger(__name__)

    def predict(self, indicators_analysis: Dict[str, Any]) -> str:
        """
        Prédit le régime de marché à partir d'une analyse d'indicateurs.
        Args:
            indicators_analysis: dict contenant les clés 'trend', 'momentum', 'volatility', 'volume', etc.
        Returns:
            str: 'bull', 'bear' ou 'range'
        """
        trend = indicators_analysis.get("trend", "neutral")
        momentum = indicators_analysis.get("momentum", 0)
        volatility = indicators_analysis.get("volatility", 0)
        volume = indicators_analysis.get("volume", "normal")

        if (
            trend == "bullish"
            and momentum > 0
            and volatility < 0.03
            and volume == "high"
        ):
            regime = "bull"
        elif (
            trend == "bearish"
            and momentum < 0
            and volatility > 0.02
            and volume == "high"
        ):
            regime = "bear"
        else:
            regime = "range"

        self.current_regime = regime
        self.logger.info(f"Régime détecté : {regime}")
        return regime

    async def study_market(
        self, market_data: pd.DataFrame, signals_func
    ) -> Dict[str, Any]:
        """
        Analyse le marché sur une période, détecte le régime et génère un rapport.
        Args:
            market_data: DataFrame des prix/volumes
            signals_func: fonction d'analyse des signaux (ex: analyze_signals du module signals)
        Returns:
            dict: rapport d'analyse détaillé
        """
        if market_data is None or len(market_data) < 50:
            self.logger.warning("Pas assez de données pour analyse du marché.")
            return {}

        # On suppose que signals_func retourne un dict de signaux
        indicators_analysis = signals_func(market_data)
        regime = self.predict(indicators_analysis)
        report = self._generate_analysis_report(indicators_analysis, regime)
        return report

    def _generate_recommendation(self, trend, momentum, volatility, volume):
        try:
            # Compteurs pour les signaux buy/sell (ancienne logique)
            buy_signals = 0
            sell_signals = 0

            # Système de points (nouvelle logique)
            points = 0

            # --- Analyse de la tendance ---
            if trend["primary_trend"] == "bullish":
                buy_signals += 1
                points += 2
            elif trend["primary_trend"] == "bearish":
                sell_signals += 1
            if trend.get("trend_strength", 0) > 25:
                points += 1
            if trend.get("trend_direction", 0) == 1:
                points += 1

            # --- Momentum ---
            if momentum.get("rsi_signal") == "oversold":
                buy_signals += 1
                points += 2
            elif momentum.get("rsi_signal") == "overbought":
                sell_signals += 1
            if momentum.get("stoch_signal") == "buy":
                points += 1
            if momentum.get("stoch_signal") == "buy":
                buy_signals += 1
            if momentum.get("stoch_signal") == "sell":
                sell_signals += 1
            if momentum.get("ultimate_signal") == "buy":
                points += 1

            # --- Volatilité ---
            if volatility.get("bb_signal") == "oversold":
                points += 1
                buy_signals += 1
            elif volatility.get("bb_signal") == "overbought":
                sell_signals += 1
            if volatility.get("kc_signal") == "breakout":
                points += 1

            # --- Volume ---
            if volume.get("mfi_signal") == "buy":
                buy_signals += 1
                points += 1
            elif volume.get("mfi_signal") == "sell":
                sell_signals += 1
            if volume.get("cmf_trend") == "positive":
                points += 1
                buy_signals += 1
            if volume.get("obv_trend") == "up":
                points += 1
                buy_signals += 1
            elif volume.get("obv_trend") == "down":
                sell_signals += 1

            # --- Génération de la recommandation finale ---
            # Par points (plus fin)
            if points >= 8:
                action = "strong_buy"
                confidence = points / 12
            elif points >= 6:
                action = "buy"
                confidence = points / 12
            elif points <= 2:
                action = "strong_sell"
                confidence = 1 - (points / 12)
            elif points <= 4:
                action = "sell"
                confidence = 1 - (points / 12)
            else:
                action = "neutral"
                confidence = 0.5

            # Par signaux purs (pour compatibilité)
            strength = abs(buy_signals - sell_signals)
            signals = {"buy": buy_signals, "sell": sell_signals}

            return {
                "action": action,
                "confidence": confidence,
                "strength": strength,
                "signals": signals,
            }

        except Exception as e:
            self.logger.error(f"❌ Erreur génération recommandation: {e}")
            return {
                "action": "error",
                "confidence": 0,
                "strength": 0,
                "signals": {"buy": 0, "sell": 0},
                "error": str(e),
            }

    def _generate_analysis_report(self, indicators_analysis, regime):
        try:
            report = f"""
╔═════════════════════════════════════════════════╗
║           RAPPORT D'ANALYSE DE MARCHÉ           ║
╠═════════════════════════════════════════════════╣    
║ Régime: {regime}                               ║
╚═════════════════════════════════════════════════╝

    📊 Analyse par Timeframe/Paire :
    """
            for timeframe, pairs_dict in indicators_analysis.items():
                for pair, analysis in pairs_dict.items():
                    report += f"""
    🕒 {timeframe} | {pair} :
    ├─ 📈 Tendance: {analysis.get('trend', {}).get('trend_strength', 'N/A')}
    ├─ 📊 Volatilité: {analysis.get('volatility', {}).get('current_volatility', 'N/A')}
    ├─ 📉 Volume: {analysis.get('volume', {}).get('volume_profile', {}).get('strength', 'N/A')}
    └─ 🎯 Signal dominant: {analysis.get('dominant_signal', 'N/A')}
    """
            return report
        except Exception as e:
            self.logger.error(f"❌ Erreur génération rapport: {e}")
            return f"Erreur lors de la génération du rapport : {e}"


class RegimeDetector:
    """Détecteur de régimes de marché"""

    def __init__(self):
        self.current_regime = None
        self.logger = logging.getLogger(__name__)

    def predict(self, indicators_analysis):
        try:
            regime = "Unknown"
            if indicators_analysis:
                trend_strength = 0
                volatility = 0
                volume = 0

                for timeframe_data in indicators_analysis.values():
                    if "trend" in timeframe_data:
                        trend_strength += timeframe_data["trend"].get(
                            "trend_strength", 0
                        )
                    if "volatility" in timeframe_data:
                        volatility += timeframe_data["volatility"].get(
                            "current_volatility", 0
                        )
                    if "volume" in timeframe_data:
                        volume += float(
                            timeframe_data["volume"]
                            .get("volume_profile", {})
                            .get("strength", 0)
                        )

                if trend_strength > 0.7:
                    regime = "Trending"
                elif volatility > 0.7:
                    regime = "Volatile"
                elif volume > 0.7:
                    regime = "High Volume"
                else:
                    regime = "Ranging"

            self.current_regime = regime
            self.logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           MARKET REGIME DETECTION                ║
╠═════════════════════════════════════════════════╣
║ Régime: {regime}
╚═════════════════════════════════════════════════╝
            """
            )
            return regime

        except Exception as e:
            self.logger.error(f"❌ Erreur détection régime: {e}")
            return "Error"


async def study_market(self, period="7d"):
    self.logger = logging.getLogger(__name__)
    self.logger.info("🔊 Étude du marché en cours...")
    if not hasattr(self, "advanced_indicators") or self.advanced_indicators is None:
        raise RuntimeError(
            "advanced_indicators non initialisé : appelle _initialize_analyzers() d'abord"
        )
    try:
        # -- Bloc critique avec logs détaillés et traceback sur erreur --
        try:
            self.logger.info("➡️ [study_market] Avant get_historical_data")
            if not getattr(self.exchange, "_initialized", False):
                self.logger.info("[study_market] Initialisation exchange...")
                await self.exchange.initialize()
            self.logger.info(
                "[study_market] Après initialize, avant get_historical_data"
            )
            get_historical = getattr(self.exchange, "get_historical_data", None)
            if asyncio.iscoroutinefunction(get_historical):
                historical_data = await get_historical(
                    self.config["TRADING"]["pairs"],
                    self.config["TRADING"]["timeframes"],
                    period,
                )
            else:
                historical_data = get_historical(
                    self.config["TRADING"]["pairs"],
                    self.config["TRADING"]["timeframes"],
                    period,
                )
            self.logger.info("⬅️ [study_market] Après get_historical_data")
        except Exception as e:
            self.logger.error(f"❌ [study_market] Exception get_historical_data: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

        if not historical_data or not isinstance(historical_data, dict):
            self.logger.error(
                "❌ Données historiques non disponibles ou mauvais format (None ou pas dict)"
            )
            raise ValueError("Données historiques non disponibles ou format inattendu")

        indicators_analysis = {}
        # Analyse sécurisée pour chaque timeframe/paire
        for timeframe in self.config["TRADING"]["timeframes"]:
            tf_data = historical_data.get(timeframe, {})
            indicators_analysis[timeframe] = {}
            for pair in self.config["TRADING"]["pairs"]:
                df = tf_data.get(pair)
                required_cols = {"open", "high", "low", "close", "volume"}
                if (
                    not isinstance(df, pd.DataFrame)
                    or df.empty
                    or not required_cols.issubset(df.columns)
                ):
                    self.logger.warning(
                        f"Données OHLCV absentes ou incomplètes pour {pair} {timeframe}, skip analyse."
                    )
                    indicators_analysis[timeframe][pair] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Aucune donnée",
                    }
                    continue
                try:
                    result = self.advanced_indicators.analyze_timeframe(df, timeframe)
                    indicators_analysis[timeframe][pair] = (
                        result
                        if result
                        else {
                            "trend": {"trend_strength": 0},
                            "volatility": {"current_volatility": 0},
                            "volume": {"volume_profile": {"strength": "N/A"}},
                            "dominant_signal": "Analyse échouée",
                        }
                    )
                except Exception as tf_error:
                    self.logger.error(f"Erreur analyse {pair} {timeframe}: {tf_error}")
                    indicators_analysis[timeframe][pair] = {
                        "trend": {"trend_strength": 0},
                        "volatility": {"current_volatility": 0},
                        "volume": {"volume_profile": {"strength": "N/A"}},
                        "dominant_signal": "Erreur",
                    }

        # Sécurise la conversion float des volumes
        for timeframe, tf_pairs in indicators_analysis.items():
            for pair, tf_analysis in tf_pairs.items():
                if (
                    "volume" in tf_analysis
                    and "volume_profile" in tf_analysis["volume"]
                ):
                    strength = tf_analysis["volume"]["volume_profile"].get(
                        "strength", 0
                    )
                    tf_analysis["volume"]["volume_profile"]["strength"] = safe_float(
                        strength, 0.0
                    )

        # Pour le calcul du régime, on peut agréger (par exemple sur le premier pair)
        regime = self.regime_detector.predict(
            {
                tf: next(iter(tf_pairs.values()), {})
                for tf, tf_pairs in indicators_analysis.items()
            }
        )
        self.logger.info(f"🔈 Régime de marché détecté: {regime}")

        try:
            analysis_report = self._generate_analysis_report(
                indicators_analysis,
                regime,
            )
            await self.telegram.send_message(analysis_report)
        except Exception as report_error:
            self.logger.error(f"Erreur génération rapport: {report_error}")

        try:
            self.dashboard.update_market_analysis(
                historical_data=historical_data,
                indicators=indicators_analysis,
                regime=regime,
            )
        except Exception as dash_error:
            self.logger.error(f"Erreur mise à jour dashboard: {dash_error}")

        return regime, historical_data, indicators_analysis

    except Exception as e:
        self.logger.error(f"Erreur study_market: {e}")
        raise
