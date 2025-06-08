
class NewsMonitor:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = BertForSequenceClassification.from_pretrained("finbert").to(self.device)
        self.sources = ["coindesk", "cointelegraph"]  # Override propre
    def __init__(self):
        self.sources = ['coindesk', 'cointelegraph']
    
    def fetch_news(self):
        return [requests.get(f'https://api.{src}.com/news').json() for src in self.sources]

    def filter_important(self, news):
        return [n for n in news if 'bitcoin' in n['title'].lower() or 'ethereum' in n['title'].lower()]
