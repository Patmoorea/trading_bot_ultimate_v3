import re
import os
import json
import shutil
from datetime import datetime


class NewsPauseManager:
    # Criticité : mot-clé associé à une durée de pause par défaut (en cycles)
    CRITICAL_KEYWORDS = {
        "hack": 30,
        "exploit": 30,
        "theft": 20,
        "attack": 15,
        "scam": 15,
        "rug": 15,
        "exit scam": 15,
        "compromised": 12,
        "security breach": 12,
        "arrest": 10,
        "frozen": 10,
        "liquidation": 10,
        "insolvency": 10,
        "lawsuit": 8,
        "investigation": 8,
        "ban": 15,
        "delist": 8,
        "paused": 6,
        "halted": 6,
        "regulation": 10,
        "suspension": 6,
    }

    def __init__(self, default_pause_cycles=5, alert_callback=None):
        """
        default_pause_cycles: durée par défaut si le mot-clé n'est pas mappé
        alert_callback: fonction appelée si un événement est détecté (ex: pour envoyer une alerte Telegram)
        """
        self.default_pause_cycles = default_pause_cycles
        self.global_cycles_remaining = 0  # Pause globale
        self.last_event_time = None
        self.last_event_news = None
        self.last_triggered_title = None
        self.alert_callback = alert_callback

        # Gestion avancée :
        self.pair_pauses = {}  # {pair: cycles_restants}
        self.buy_paused_pairs = set()  # Paires où seuls les achats sont bloqués
        self.active_pauses = []

        self.volatility_thresholds = {"low": 0.02, "medium": 0.05, "high": 0.08}
        self.market_conditions = {}

    def smart_pause_update(self, bot):
        """
        Gère dynamiquement la levée ou le maintien des pauses selon plusieurs critères :
        - Signaux techniques/IA
        - Sentiment news
        - Volatilité/volume
        - Prix qui repart fortement
        - Pause partielle : autorise les ventes, bloque les achats
        """
        # 1. Déblocage anticipé si le marché repart fortement à la hausse
        regime = getattr(bot, "regime", None)
        sentiment = None
        try:
            sentiment = bot.get_performance_metrics().get("sentiment", 0)
        except Exception:
            pass

        # Critère 1 : Marché haussier puissant
        if regime == "TRENDING_UP":
            if self.global_cycles_remaining > 2:
                print("[SMART PAUSE] Marché haussier, réduction de la pause globale !")
                self.global_cycles_remaining = max(self.global_cycles_remaining // 2, 1)

        # Critère 2 : Sentiment news positif
        if sentiment is not None and sentiment > 0.5:
            print(
                "[SMART PAUSE] Sentiment news positif, réduction de la pause globale !"
            )
            self.global_cycles_remaining = max(self.global_cycles_remaining // 2, 1)

        # Critère 3 : Volatilité/Volume normalisés
        for pair, cycles in list(self.pair_pauses.items()):
            vol = bot.calculate_volatility(bot.market_data.get(pair, {}).get("1h", {}))
            avg_vol = bot.calculate_volume_profile(
                bot.market_data.get(pair, {}).get("1h", {})
            ).get("strength", 1)
            if vol < 0.05 and avg_vol > 0.7:
                print(f"[SMART PAUSE] Volatilité/volume OK sur {pair}, pause réduite !")
                self.pair_pauses[pair] = max(cycles // 2, 1)

        # Critère 4 : Si le prix monte > 7% pendant la pause, lève la pause sur la paire
        for pair, cycles in list(self.pair_pauses.items()):
            market_data = bot.market_data.get(pair, {}).get("1h", {})
            if "close" in market_data and len(market_data["close"]) > 10:
                prices = market_data["close"][-10:]
                if prices[-1] > prices[0] * 1.07:  # +7% sur la période de pause
                    print(
                        f"[SMART PAUSE] Prix {pair} +7% pendant la pause, pause levée !"
                    )
                    self.pair_pauses[pair] = 0

        # Critère 5 : Maintien pause stricte si news de hack/scam/frozen
        # (optionnel)

        # Critère 6 : Pause partielle (autorise les ventes, bloque les achats)
        # géré dans la boucle principale (voir ci-dessous)

    def safe_update_shared_data(
        self, new_fields: dict, data_file="src/shared_data.json"
    ):
        # 1. Lis le fichier existant SANS jamais repartir sur {}
        try:
            with open(data_file, "r") as f:
                shared_data = json.load(f)
        except Exception:
            # En cas de bug, tente de restaurer une sauvegarde précédente
            backup_file = data_file + ".bak"
            if os.path.exists(backup_file):
                with open(backup_file, "r") as f:
                    shared_data = json.load(f)
            else:
                shared_data = None
        # Si shared_data est None, NE PAS ÉCRIRE !
        if shared_data is None:
            print("[SAFE PATCH] shared_data.json corrompu, skip écriture !")
            return
        # 2. Mets à jour les champs nécessaires
        shared_data.update(new_fields)
        # 3. Sauvegarde une copie de secours avant d’écrire
        try:
            shutil.copyfile(data_file, data_file + ".bak")
        except Exception:
            pass
        # 4. Écris
        with open(data_file, "w") as f:
            json.dump(shared_data, f, indent=4)

    def activate_pause(self, pause_decision):
        # Ajoute la pause active dans la RAM ou le fichier partagé
        self.active_pauses.append(pause_decision)

    def scan_news(self, news_list):
        """
        Scanne les news et déclenche une pause globale ou ciblée si un mot-clé critique est détecté.
        news_list: liste de dicts (doit contenir "title", "text", et idéalement "symbols"/"assets"/"sentiment")
        """
        # Debug initial
        print(f"\n[NEWSPAUSE DEBUG] Analyse de {len(news_list)} news")

        if not news_list:
            print("[NEWSPAUSE DEBUG] Liste de news vide")
            return False

        triggered = False
        for i, news in enumerate(news_list):
            # Debug de chaque news
            print(f"\n[NEWSPAUSE DEBUG] Analyse news #{i+1}:")
            print(f"- Title: {news.get('title', 'NO TITLE')}")
            print(f"- Text: {news.get('text', 'NO TEXT')[:100]}...")
            print(f"- Sentiment: {news.get('sentiment', 'NO SENTIMENT')}")
            print(f"- Symbols: {news.get('symbols', []) or news.get('assets', [])}")
            print(f"- Processed: {news.get('processed', False)}")

            # Si déjà traitée, skip
            if news.get("processed"):
                print("➡️ News déjà traitée, skip")
                continue

            title = news.get("title", "") or ""
            text = news.get("text", "") or ""
            content = f"{title} {text}".lower()
            symbols = news.get("symbols", []) or news.get("assets", [])
            sentiment = float(news.get("sentiment", 0)) if "sentiment" in news else None

            # Debug des mots-clés recherchés
            print("\nRecherche des mots-clés critiques:")
            for keyword, pause_cycles in self.CRITICAL_KEYWORDS.items():
                if re.search(rf"\b{re.escape(keyword)}\b", content):
                    print(f"⚠️ Mot-clé '{keyword}' trouvé!")
                    if title == self.last_triggered_title:
                        print("➡️ Même titre que précédent, skip")
                        continue

                    # Calcul durée pause
                    cycles = pause_cycles
                    if sentiment is not None and abs(sentiment) > 0.7:
                        cycles = int(cycles * 1.5)
                        print(f"Durée augmentée (sentiment fort): {cycles}")
                    elif sentiment is not None and abs(sentiment) < 0.3:
                        cycles = int(max(2, cycles * 0.5))
                        print(f"Durée réduite (sentiment faible): {cycles}")

                    news["processed"] = True

                    if symbols:
                        for sym in symbols:
                            if keyword in [
                                "regulation",
                                "lawsuit",
                                "investigation",
                                "ban",
                            ]:
                                self.buy_paused_pairs.add(sym)
                                self.pair_pauses[sym] = cycles
                                print(f"🔒 BUY PAUSE {cycles} cycles pour {sym}")
                            else:
                                self.pair_pauses[sym] = cycles
                                print(f"🔒 FULL PAUSE {cycles} cycles pour {sym}")
                    else:
                        self.global_cycles_remaining = cycles
                        print(f"🔒 PAUSE GLOBALE {cycles} cycles")

                    self.last_event_time = datetime.now()
                    self.last_event_news = news
                    self.last_triggered_title = title

                    if self.alert_callback:
                        self.alert_callback(keyword, news)

                    triggered = True
                    break  # Sort de la boucle des mots-clés si un est trouvé

        print(
            f"\n[NEWSPAUSE DEBUG] Résultat final: {'⚠️ Pause activée' if triggered else '✅ Aucune pause nécessaire'}"
        )
        return triggered

    def should_pause(self, news_item, market_data):
        """
        Décide automatiquement si une pause doit être déclenchée selon :
        - sentiment négatif
        - impact élevé
        - classification automatique
        - réaction du marché (hausse de volatilité)
        - propagation multi-sources
        """
        sentiment = news_item.get("sentiment", 0)
        impact = news_item.get("impact_score", 0)
        risk_class = news_item.get("risk_class", "")
        n_sources = news_item.get("n_sources", 1)
        symbol = news_item.get("symbols", ["GLOBAL"])[0]
        vol_before = market_data.get(symbol, {}).get("volatility", 0)
        vol_after = market_data.get(symbol, {}).get("volatility_post_news", vol_before)

        # Pause totale si risque systémique
        if risk_class in ["Réglementaire", "Hack"] and impact > 0.6:
            return {
                "type": "total",
                "reason": news_item.get("title"),
                "duration": 10,
                "pair": symbol,
            }

        # Pause sur la paire si sentiment très négatif ET hausse de volatilité
        if sentiment < -0.5 and vol_after > vol_before * 2:
            return {
                "type": "pair",
                "pair": symbol,
                "reason": news_item.get("title"),
                "duration": 5,
            }

        # Pause si la même news repérée sur >2 sources et impact élevé
        if n_sources >= 2 and impact > 0.5:
            return {
                "type": "total",
                "reason": news_item.get("title"),
                "duration": 10,
                "pair": symbol,
            }

        # Pause sur les shorts si news "Short squeeze"
        if "short squeeze" in news_item.get("title", "").lower():
            return {
                "type": "short_only",
                "reason": news_item.get("title"),
                "duration": 3,
                "pair": symbol,
            }

        return None  # Pas de pause

    def on_cycle_end(self):
        """
        Gestion synchronisée de la décrémentation des compteurs
        """
        print(
            "[NEWSPAUSE] Avant decrement:",
            self.global_cycles_remaining,
            self.pair_pauses,
        )

        # Liste des pauses à retirer
        to_remove = []

        # 1. Décrémentation synchronisée
        if self.global_cycles_remaining > 0:
            self.global_cycles_remaining -= 1

        for pair, cycles in list(self.pair_pauses.items()):
            if cycles > 0:
                self.pair_pauses[pair] = max(0, cycles - 1)
            if self.pair_pauses[pair] <= 0:
                to_remove.append(pair)

        # 2. Nettoyage des pauses terminées
        for pair in to_remove:
            self.pair_pauses.pop(pair, None)
            self.buy_paused_pairs.discard(pair)

        # 3. Mise à jour du fichier shared_data.json
        try:
            current_pauses = self.get_active_pauses()
            self.safe_update_shared_data(
                {
                    "active_pauses": current_pauses,
                    "pause_status": {
                        "global_remaining": self.global_cycles_remaining,
                        "pair_pauses": self.pair_pauses,
                        "max_remaining": max(
                            [self.global_cycles_remaining]
                            + list(self.pair_pauses.values()),
                            default=0,
                        ),
                    },
                }
            )
        except Exception as e:
            print(f"[ERROR] Erreur mise à jour pauses: {e}")

        print(
            "[NEWSPAUSE] Après decrement:",
            self.global_cycles_remaining,
            self.pair_pauses,
        )

    def get_last_event(self):
        """Retourne la dernière news critique détectée"""
        return self.last_event_news

    def reset(self):
        """Réinitialise toutes les pauses manuellement"""
        self.global_cycles_remaining = 0
        self.pair_pauses.clear()
        self.buy_paused_pairs.clear()
        self.last_event_news = None
        self.last_event_time = None
        self.last_triggered_title = None

    def get_active_pauses(self):
        """
        Retourne la liste des pauses actives avec gestion cohérente des compteurs.
        """
        pauses = []
        # Calcul du max des cycles pour une cohérence globale
        max_cycles = max(
            [self.global_cycles_remaining] + list(self.pair_pauses.values()), default=0
        )

        # Pauses par paire (ciblées)
        for pair, cycles_left in self.pair_pauses.items():
            if cycles_left > 0:
                pause_type = "BUY" if pair in self.buy_paused_pairs else "FULL"
                reason = getattr(self, "last_event_news", {}).get("title", "")
                pauses.append(
                    {
                        "asset": pair,
                        "action": pause_type,
                        "cycles_left": cycles_left,
                        "type": pause_type,
                        "reason": reason,
                        "max_cycles": max_cycles,  # Ajout pour cohérence
                    }
                )

        # Pause globale
        if self.global_cycles_remaining > 0:
            reason = getattr(self, "last_event_news", {}).get("title", "")
            pauses.append(
                {
                    "asset": "GLOBAL",
                    "action": "ALL",
                    "cycles_left": self.global_cycles_remaining,
                    "type": "GLOBAL",
                    "reason": reason,
                    "max_cycles": max_cycles,  # Ajout pour cohérence
                }
            )

        # Ajout information de synchronisation
        if pauses:
            for pause in pauses:
                pause["total_remaining"] = max_cycles

        return pauses

    def analyze_market_conditions(self, price_data, volume_data):
        """
        Analyse avancée des conditions de marché pour un meilleur timing
        """
        for symbol in price_data:
            volatility = self.calculate_rolling_volatility(price_data[symbol])
            volume_profile = self.analyze_volume_profile(volume_data[symbol])
            momentum = self.calculate_momentum_score(price_data[symbol])

            self.market_conditions[symbol] = {
                "volatility": volatility,
                "volume_profile": volume_profile,
                "momentum": momentum,
            }

    def should_enter_trade(self, symbol):
        """
        Vérifie si les conditions sont optimales pour entrer
        """
        conditions = self.market_conditions.get(symbol, {})

        if (
            conditions.get("volatility", 1) < self.volatility_thresholds["medium"]
            and conditions.get("volume_profile", 0) > 0.7
            and conditions.get("momentum", 0) > 0
        ):
            return True

        return False
