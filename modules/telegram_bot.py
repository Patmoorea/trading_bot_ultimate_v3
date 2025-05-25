def send_enhanced_alert(symbol, action, reason,
                        confidence, timeframe, news_link=None):
    message = f"""
    🚨 {action} Signal 🚨
    Pair: {symbol}
    Reason: {reason}
    Confidence: {confidence:.2f}/1.0
    Timeframe: {timeframe}
    """
    if news_link:
        message += f"\n📰 Related News: {news_link}"

    bot.send_message(message)

class EnhancedTelegramBot:
    def __init__(self):
        self.bot = TelegramBot()
    
    def send_market_alert(self, news_item, market_context):
        """Version enrichie avec contexte marché"""
        msg = f"""
        📰 {news_item.get('source', 'Unknown')} 
        🚨 Impact: {news_item.get('sentiment', 0):.2f} 
        📊 Trend: {market_context.get('trend', 'N/A')} 
        🔍 Volatility: {market_context.get('volatility', 0):.2f}%
        """
        self.bot.send_enhanced_alert(
            symbol=news_item.get('symbol', 'GLOBAL'),
            action="ALERT",
            reason=news_item.get('title', 'Market News'),
            confidence=abs(float(news_item.get('sentiment', 0))),
            timeframe="REALTIME",
            news_link=news_item.get('url')
        )
