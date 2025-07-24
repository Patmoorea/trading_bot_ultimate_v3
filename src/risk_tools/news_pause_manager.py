import re
import time

TRIGGER_WORDS = [
    "hack", "exploit", "ban", "scam", "rug", "rugpull", "delist", "regulation",
    "liquidation", "fraud", "lawsuit", "security breach", "stolen", "arrest",
    "shutdown", "cease", "stop trading"
]

class NewsPauseManager:
    def __init__(self, cooldown_cycles=6):
        self.cooldown_cycles = cooldown_cycles
        self.pause_until_cycle = 0
        self.last_triggered_news = []

    def check_news_and_trigger(self, news_list, current_cycle):
        """
        Parcourt la liste des news (dicts avec 'title') et active le mode pause si un trigger est détecté.
        """
        for news in news_list:
            title = news.get("title", "").lower()
            for word in TRIGGER_WORDS:
                if re.search(fr"\b{word}\b", title):
                    if title not in self.last_triggered_news:
                        self.pause_until_cycle = current_cycle + self.cooldown_cycles
                        self.last_triggered_news.append(title)
                        print(f"[NEWS PAUSE] Pause trading déclenchée à cause de la news: {title}")
                        return True  # Pause déclenchée
        return False  # Rien de déclenché

    def is_paused(self, current_cycle):
        """Retourne True si le bot doit être en pause à ce cycle."""
        return current_cycle < self.pause_until_cycle