import re
from datetime import datetime


class NewsPauseManager:
    # Criticité : mot-clé associé à une durée de pause par défaut (en cycles)
    CRITICAL_KEYWORDS = {
        "hack": 20,
        "exploit": 20,
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
        "ban": 8,
        "delist": 8,
        "paused": 6,
        "halted": 6,
        "regulation": 6,
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

    def safe_update_shared_data(new_fields: dict, data_file="src/shared_data.json"):
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
        triggered = False
        for news in news_list:
            title = news.get("title", "") or ""
            text = news.get("text", "") or ""
            content = f"{title} {text}".lower()
            # Extraction symboles/paire concernés si possible
            symbols = news.get("symbols", []) or news.get("assets", [])
            # Score de sentiment éventuel (pour adaptation)
            sentiment = float(news.get("sentiment", 0)) if "sentiment" in news else None

            for keyword, pause_cycles in self.CRITICAL_KEYWORDS.items():
                if re.search(rf"\b{re.escape(keyword)}\b", content):
                    if title == self.last_triggered_title:
                        continue  # Ne pas relancer sur la même news

                    # Durée de pause ajustée par criticité et sentiment
                    cycles = pause_cycles
                    if sentiment is not None and abs(sentiment) > 0.7:
                        cycles = int(cycles * 1.5)
                    elif sentiment is not None and abs(sentiment) < 0.3:
                        cycles = int(max(2, cycles * 0.5))

                    # PATCH FONDAMENTAL : Marquer la news comme traitée DÈS le déclenchement
                    news["processed"] = True

                    # Pause ciblée si symboles détectés, sinon globale
                    if symbols:
                        for sym in symbols:
                            # Pour certains mots-clés, on bloque uniquement les achats (BUY)
                            if keyword in [
                                "regulation",
                                "lawsuit",
                                "investigation",
                                "ban",
                            ]:
                                self.buy_paused_pairs.add(sym)
                                self.pair_pauses[sym] = cycles
                                print(
                                    f"[NEWS PAUSE] Trigger: '{keyword}' -> BUY PAUSE {cycles} cycles for {sym} (news: {title[:80]})"
                                )
                            else:
                                self.pair_pauses[sym] = cycles
                                print(
                                    f"[NEWS PAUSE] Trigger: '{keyword}' -> FULL PAUSE {cycles} cycles for {sym} (news: {title[:80]})"
                                )
                    else:
                        # Pas de symboles → pause globale
                        self.global_cycles_remaining = cycles
                        print(
                            f"[NEWS PAUSE] Trigger: '{keyword}' -> GLOBAL PAUSE {cycles} cycles (news: {title[:80]})"
                        )

                    self.last_event_time = datetime.now()
                    self.last_event_news = news
                    self.last_triggered_title = title
                    if self.alert_callback:
                        self.alert_callback(keyword, news)
                    triggered = True
        return triggered

    def should_pause(news_item, market_data):
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
            return {"type": "total", "reason": news_item.get("title"), "duration": 10}

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
            return {"type": "total", "reason": news_item.get("title"), "duration": 10}

        # Pause sur les shorts si news "Short squeeze"
        if "short squeeze" in news_item.get("title", "").lower():
            return {
                "type": "short_only",
                "reason": news_item.get("title"),
                "duration": 3,
            }

        return None  # Pas de pause

    def on_cycle_end(self):
        """
        Doit être appelée à chaque tick/cycle.
        Décrémente les compteurs de pause (globale et par paire).
        Nettoie les pauses terminées.
        """
        print(
            "[NEWSPAUSE] Avant decrement:",
            self.global_cycles_remaining,
            self.pair_pauses,
        )
        # Décrémentation pause globale
        if self.global_cycles_remaining > 0:
            self.global_cycles_remaining -= 1

        # Décrémentation des pauses par paire
        to_remove = []
        for pair, cycles in list(self.pair_pauses.items()):
            if cycles > 0:
                self.pair_pauses[pair] -= 1
            if self.pair_pauses[pair] <= 0:
                to_remove.append(pair)
        for pair in to_remove:
            self.pair_pauses.pop(pair, None)
            self.buy_paused_pairs.discard(pair)
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
        Retourne la liste des pauses actives sous forme de dicts.
        Exemple : [{"asset": "ETH", "action": "BUY", "cycles_left": 8, "type": "FULL"}, ...]
        """
        pauses = []
        # Pauses par paire (ciblées)
        for pair, cycles_left in self.pair_pauses.items():
            if cycles_left > 0:
                pause_type = "BUY" if pair in self.buy_paused_pairs else "FULL"
                pauses.append(
                    {
                        "asset": pair,
                        "action": pause_type,
                        "cycles_left": cycles_left,
                        "type": pause_type,
                    }
                )
        # Pause globale
        if self.global_cycles_remaining > 0:
            pauses.append(
                {
                    "asset": "GLOBAL",
                    "action": "ALL",
                    "cycles_left": self.global_cycles_remaining,
                    "type": "GLOBAL",
                }
            )
        return pauses
