import os


class NewsAPI:
    """Version totalement autonome"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY", "demo")

    def fetch_news(self):
        return {"status": "OK", "api_key": bool(self.api_key)}
