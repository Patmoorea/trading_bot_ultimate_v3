import re
from datetime import datetime


class NewsPauseManager:
    CRITICAL_KEYWORDS = [
        "hack",
        "exploit",
        "ban",
        "delist",
        "scam",
        "rug",
        "exit scam",
        "theft",
        "attack",
        "paused",
        "halted",
        "investigation",
        "compromised",
        "lawsuit",
        "liquidation",
        "insolvency",
        "regulation",
        "frozen",
        "arrest",
        "suspension",
        "security breach",
    ]

    def __init__(self, pause_cycles=5, alert_callback=None):
        """
        pause_cycles: nombre de cycles à mettre en pause si un événement critique est détecté
        alert_callback: fonction appelée si un événement est détecté (ex: pour envoyer une alerte Telegram)
        """
        self.pause_cycles = pause_cycles
        self.cycles_remaining = 0
        self.last_event_time = None
        self.last_event_news = None
        self.last_triggered_title = None
        self.alert_callback = alert_callback

    def scan_news(self, news_list):
        """
        Scanne la liste des news et déclenche une pause si un mot-clé critique est détecté.
        news_list: liste de dicts (doit contenir "title" et "text")
        """
        # N'analyse que si la pause n'est pas déjà active
        if self.cycles_remaining > 0:
            return False
        for news in news_list:
            title = news.get("title", "") or ""
            text = news.get("text", "") or ""
            content = f"{title} {text}".lower()
            for keyword in self.CRITICAL_KEYWORDS:
                if re.search(rf"\b{re.escape(keyword)}\b", content):
                    # Ne relance pas la pause sur la même news déjà traitée
                    if title == self.last_triggered_title:
                        continue
                    self.cycles_remaining = self.pause_cycles
                    self.last_event_time = datetime.now()
                    self.last_event_news = news
                    self.last_triggered_title = title
                    print(
                        f"[NEWS PAUSE] Trigger: '{keyword}' detected in news: {title[:120]}"
                    )
                    if self.alert_callback:
                        self.alert_callback(keyword, news)
                    return True  # Pause déclenchée
        return False  # Pas d'événement critique

    def should_pause(self):
        """Retourne True si le trading doit être en pause"""
        return self.cycles_remaining > 0

    def on_cycle_end(self):
        """À appeler en fin de chaque cycle de trading pour décrémenter la pause"""
        if self.cycles_remaining > 0:
            self.cycles_remaining -= 1

    def get_last_event(self):
        """Retourne la dernière news critique détectée"""
        return self.last_event_news

    def reset(self):
        """Réinitialise la pause manuellement"""
        self.cycles_remaining = 0
        self.last_event_news = None
        self.last_event_time = None
        self.last_triggered_title = None
