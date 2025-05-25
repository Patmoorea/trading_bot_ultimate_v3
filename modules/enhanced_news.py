import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()


class NewsAnalyzer:
    def __init__(self):
        self.sources = os.getenv("NEWS_SOURCES", "").split(",")
        self.languages = os.getenv("NEWS_API_LANGUAGES", "en").split(",")

    def should_alert(self, news_item):
        """Détermine si une news mérite une alerte"""
        return any(
            source in news_item.get("source", {}).get("id", "")
            for source in self.sources
        )
