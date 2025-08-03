class EnhancedRiskManager:
    def __init__(self):
        # Limites et seuils
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.position_limits = {
            "max_per_trade": 0.05,  # 5% max par trade
            "max_total_exposure": 0.25,  # 25% max exposition totale
        }
        self.min_confidence = 0.8  # 80% confiance minimum
        self.correlation_threshold = 0.7  # Seuil corrélation max

        # Seuils de validation
        self.validation_thresholds = {
            "technical": 0.3,  # Score technique minimum
            "momentum": 0.2,  # Score momentum minimum
            "orderflow": 0.2,  # Score orderflow minimum
            "liquidity": 0.7,  # Seuil liquidité maximum
            "pressure": 0.8,  # Seuil pression marché maximum
        }

    def validate_trade(self, signals):
        """Validation complète d'un trade selon tous les critères"""
        try:
            if not signals or not isinstance(signals, dict):
                print("[RISK] Signaux invalides")
                return False

            # Extraction des scores
            technical = signals.get("technical", {})
            momentum = signals.get("momentum", {})
            orderflow = signals.get("orderflow", {})

            if not all([technical, momentum, orderflow]):
                print("[RISK] Signaux incomplets")
                return False

            # Scores principaux
            tech_score = float(technical.get("score", 0))
            momentum_score = float(momentum.get("score", 0))
            flow_score = float(orderflow.get("score", 0))

            # Score global pondéré
            weights = {"technical": 0.4, "momentum": 0.3, "orderflow": 0.3}
            total_score = (
                tech_score * weights["technical"]
                + momentum_score * weights["momentum"]
                + flow_score * weights["orderflow"]
            )

            # === VALIDATION DES CRITÈRES ===

            # 1. Score technique
            if abs(tech_score) < self.validation_thresholds["technical"]:
                print(f"[RISK] Score technique insuffisant: {tech_score:.2f}")
                return False

            # 2. Momentum
            if abs(momentum_score) < self.validation_thresholds["momentum"]:
                print(f"[RISK] Momentum faible: {momentum_score:.2f}")
                return False

            # 3. Orderflow
            if abs(flow_score) < self.validation_thresholds["orderflow"]:
                print(f"[RISK] Orderflow insuffisant: {flow_score:.2f}")
                return False

            # 4. Liquidité
            liquidity = float(orderflow.get("liquidity", 0))
            if abs(liquidity) > self.validation_thresholds["liquidity"]:
                print(f"[RISK] Liquidité anormale: {liquidity:.2f}")
                return False

            # 5. Pression marché
            pressure = float(orderflow.get("market_pressure", 0))
            if abs(pressure) > self.validation_thresholds["pressure"]:
                print(f"[RISK] Pression marché excessive: {pressure:.2f}")
                return False

            # Score global minimum
            if abs(total_score) < 0.25:
                print(f"[RISK] Score global insuffisant: {total_score:.2f}")
                return False

            print(f"[RISK] ✅ Trade validé - Score: {total_score:.2f}")
            return True

        except Exception as e:
            print(f"[RISK] Erreur validation: {e}")
            return False

    def calculate_position_size(self, equity, confidence, volatility, correlation):
        """Calcul intelligent de la taille de position"""
        try:
            # Vérification confiance minimum
            if confidence < self.min_confidence:
                print(f"[RISK] Confiance insuffisante: {confidence:.2f}")
                return 0

            # Taille de base
            base_size = equity * self.position_limits["max_per_trade"]

            # Ajustements
            vol_adj = max(0.3, 1 - (volatility * 2))  # Réduction si volatilité élevée
            corr_adj = max(0.3, 1 - correlation)  # Réduction si corrélation élevée

            # Application des ajustements
            size = base_size * vol_adj * corr_adj

            # Respect de la limite max par trade
            final_size = min(size, equity * self.position_limits["max_per_trade"])

            print(
                f"[RISK] Taille calculée: {final_size:.2f} (vol_adj={vol_adj:.2f}, corr_adj={corr_adj:.2f})"
            )
            return final_size

        except Exception as e:
            print(f"[RISK] Erreur calcul position: {e}")
            return 0

    def check_exposure_limit(self, current_positions, new_position_size):
        """Vérification des limites d'exposition"""
        try:
            # Calcul exposition totale
            total_exposure = sum(pos["size"] for pos in current_positions.values())
            new_total = total_exposure + new_position_size

            # Vérification limite
            is_valid = new_total <= self.position_limits["max_total_exposure"]

            print(f"[RISK] Exposition: {new_total:.2%} {'✅' if is_valid else '❌'}")
            return is_valid

        except Exception as e:
            print(f"[RISK] Erreur vérification exposition: {e}")
            return False

    def calculate_drawdown(self, equity_curve):
        """Calcul du drawdown actuel"""
        try:
            if not equity_curve:
                return 0

            peak = max(equity_curve)
            current = equity_curve[-1]

            if peak == 0:
                return 0

            drawdown = (current - peak) / peak
            print(f"[RISK] Drawdown actuel: {drawdown:.2%}")

            return abs(drawdown)

        except Exception as e:
            print(f"[RISK] Erreur calcul drawdown: {e}")
            return 0

    def adjust_for_market_conditions(self, base_size, market_regime):
        """Ajustement selon conditions de marché"""
        try:
            # Ajustements par régime
            regime_multipliers = {
                "TRENDING_UP": 1.0,  # Normal en tendance haussière
                "TRENDING_DOWN": 0.7,  # Réduction en tendance baissière
                "RANGING": 0.8,  # Réduction en range
                "VOLATILE": 0.5,  # Forte réduction en volatilité
            }

            multiplier = regime_multipliers.get(market_regime, 0.7)
            adjusted_size = base_size * multiplier

            print(f"[RISK] Ajustement régime {market_regime}: {multiplier:.2f}x")
            return adjusted_size

        except Exception as e:
            print(f"[RISK] Erreur ajustement marché: {e}")
            return base_size * 0.7  # Réduction par défaut
