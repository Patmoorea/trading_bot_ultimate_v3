class NewsImpactEvaluator:
    IMPACT_LEVELS = {
        'high': ['war', 'attack', 'hack', 'crash'],
        'medium': ['regulation', 'law', 'ban'],
        'low': ['partnership', 'update', 'release']
    }
    
    def evaluate_impact(self, news_text):
        text_lower = news_text.lower()
        for level, keywords in self.IMPACT_LEVELS.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        return 'none'
