class EnhancedTelegramBot:
    def send_market_alert(self, news_item, market_context):
        """Format enrichi avec contexte marché"""
        msg = f"""
        📰 {news_item['source']} 
        🚨 Impact: {news_item['sentiment']} 
        📊 Market: {market_context['trend']} 
        🔍 Volatility: {market_context['volatility']}%
        """
        self._send(msg)
