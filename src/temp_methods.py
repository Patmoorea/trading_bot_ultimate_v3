    def _generate_analysis_report(self, indicators_analysis, regime, news_sentiment=None):
        """Génère un rapport d'analyse détaillé avec news"""
        current_time = "2025-06-01 00:28:15"  # Mise à jour
        
        report = [
            "📊 Analyse complète du marché:",
            f"Date: {current_time} UTC",
            f"Trader: Patmoorea",
            f"Régime: {regime}",
            "\nTendances principales:"
        ]
        
        # Ajout de l'analyse des news si disponible
        if news_sentiment:
            try:
                report.extend([
                    "\n📰 Analyse des News:",
                    f"Sentiment: {news_sentiment.get('overall_sentiment', 0):.2%}",
                    f"Impact estimé: {news_sentiment.get('impact_score', 0):.2%}",
                    f"Événements majeurs: {news_sentiment.get('major_events', 'Aucun')}"
                ])
            except Exception as e:
                logger.warning(f"Erreur traitement news: {e}")
        
        # Analyse par timeframe
        for timeframe, analysis in indicators_analysis.items():
            try:
                report.append(f"\n⏰ {timeframe}:")
                trend_strength = analysis.get('trend', {}).get('trend_strength', 0)
                volatility = analysis.get('volatility', {}).get('current_volatility', 0)
                volume_profile = analysis.get('volume', {}).get('volume_profile', {})
                
                report.extend([
                    f"- Force de la tendance: {trend_strength:.2%}",
                    f"- Volatilité: {volatility:.2%}",
                    f"- Volume: {volume_profile.get('strength', 'N/A')}",
                    f"- Signal dominant: {analysis.get('dominant_signal', 'Neutre')}"
                ])
            except Exception as e:
                logger.warning(f"Erreur analyse timeframe {timeframe}: {e}")
                report.extend([
                    f"\n⏰ {timeframe}:",
                    "- Données non disponibles",
                    "- Analyse en cours..."
                ])
        
        return "\n".join(report)

    async def process_market_data(self):
        """Traitement des données de marché avec tous les indicateurs"""
        try:
            # Récupération des données
            market_data = self.buffer.get_latest()
            
            # Vérification des données
            if market_data is None or not market_data:
                logger.warning("Données de marché manquantes")
                return None, None
            
            # Calcul de tous les indicateurs
            indicators_results = {}
            for timeframe in config["TRADING"]["timeframes"]:
                try:
                    tf_data = market_data[timeframe]
                    indicators_results[timeframe] = self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
                except Exception as e:
                    logger.error(f"Erreur calcul indicateurs {timeframe}: {e}")
                    indicators_results[timeframe] = {
                        'trend': {'trend_strength': 0},
                        'volatility': {'current_volatility': 0},
                        'volume': {'volume_profile': {'strength': 'N/A'}},
                        'dominant_signal': 'Neutre'
                    }
            
            # Génération de la heatmap de liquidité
            try:
                orderbook = await self.exchange.get_orderbook(config["TRADING"]["pairs"])
                heatmap = self.generate_heatmap(orderbook)
            except Exception as e:
                logger.error(f"Erreur génération heatmap: {e}")
                heatmap = None
            
            # Notification des signaux importants
            await self._notify_significant_signals(indicators_results)
            
            # Mise à jour du dashboard en temps réel
            self.dashboard.update(
                market_data,
                indicators_results,
                heatmap,
                current_time="2025-06-01 00:28:15"
            )
            
            return market_data, indicators_results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {e}")
            await self.telegram.send_message(f"⚠️ Erreur traitement: {str(e)}")
            return None, None

    async def analyze_signals(self, market_data, indicators):
        """Analyse technique et fondamentale avancée"""
        try:
            # Vérification des données
            if market_data is None or indicators is None:
                logger.warning("Données manquantes pour l'analyse")
                return None

            # Analyse technique
            try:
                technical_features = self.hybrid_model.analyze_technical(
                    market_data=market_data,
                    indicators=indicators
                )
                
                if not isinstance(technical_features, dict):
                    technical_features = {
                        'tensor': technical_features,
                        'score': float(torch.mean(technical_features).item())
                    }
            except Exception as e:
                logger.error(f"Erreur analyse technique: {e}")
                technical_features = {'score': 0.5, 'tensor': None}

            # Analyse des news
            try:
                news_impact = await self.news_analyzer.analyze_recent_news()
            except Exception as e:
                logger.warning(f"Erreur analyse news: {e}")
                news_impact = {
                    'sentiment': {'score': 0.5},
                    'impact': 0,
                    'summary': "Pas de données news disponibles"
                }
            
            try:
                current_regime = self.regime_detector.detect_regime(indicators)
            except Exception as e:
                logger.error(f"Erreur détection régime: {e}")
                current_regime = "Indéterminé"
            
            # Construction de la décision finale
            try:
                combined_features = self._combine_features(
                    technical_features,
                    news_impact,
                    current_regime
                )
                
                policy, value = self.decision_model(combined_features)
                
                decision = self._build_decision(
                    policy=policy,
                    value=value,
                    technical_score=technical_features['score'],
                    news_sentiment=news_impact['sentiment'],
                    regime=current_regime
                )
                
                decision = self._add_risk_management(decision)
                
                logger.info(
                    f"Décision générée - Action: {decision['action']}, "
                    f"Confiance: {decision['confidence']:.2%}, "
                    f"Régime: {decision['regime']}"
                )
                
                return decision
                
            except Exception as e:
                logger.error(f"Erreur construction décision: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Erreur analyse signaux: {e}")
            await self.telegram.send_message(f"⚠️ Erreur analyse: {str(e)}")
            return None
