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

    def should_pause(self, pair=None, action="ALL"):
        """
        Retourne True si le trading doit être en pause :
        - pair: optionnel, si précisé, ne bloque que cette paire
        - action: 'ALL' (tout), 'BUY' (seulement les achats), 'SELL' (seulement les ventes)
        """
        # Pause globale stricte
        if self.global_cycles_remaining > 0:
            return True

        # Pause ciblée sur une paire
        if pair:
            pause = self.pair_pauses.get(pair, 0)
            if pause > 0:
                if action == "BUY" and pair in self.buy_paused_pairs:
                    return True
                if action == "ALL" and pair not in self.buy_paused_pairs:
                    return True
        return False

    def on_cycle_end(self):
        print(
            "[NEWSPAUSE] Avant decrement:",
            self.global_cycles_remaining,
            self.pair_pauses,
        )
        if self.global_cycles_remaining > 0:
            self.global_cycles_remaining -= 1
        to_remove = []
        for pair, cycles in self.pair_pauses.items():
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
