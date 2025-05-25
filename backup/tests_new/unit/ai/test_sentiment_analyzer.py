import pytest
from src.ai.sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

class TestSentimentAnalyzer:
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert isinstance(analyzer.sources, dict)
        assert all(source in analyzer.sources for source in ['twitter', 'reddit', 'news'])
        
    def test_analyze_text(self, analyzer):
        """Test single text analysis"""
        # Test positive sentiment
        text_pos = "Bitcoin is showing bullish momentum"
        result_pos = analyzer.analyze_text(text_pos)
        assert result_pos['sentiment'] == 'positive'
        
        # Test negative sentiment
        text_neg = "Market is dropping"
        result_neg = analyzer.analyze_text(text_neg)
        assert result_neg['sentiment'] == 'negative'
        
    def test_analyze_batch(self, analyzer):
        """Test batch text analysis"""
        texts = [
            "Bullish on Bitcoin",
            "Market looks weak",
            "Nothing special happening"
        ]
        results = analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all(k in r for r in results for k in ['sentiment', 'score', 'polarity', 'subjectivity'])
        
    def test_aggregated_sentiment(self, analyzer):
        """Test aggregated sentiment analysis"""
        result = analyzer.get_aggregated_sentiment("BTC/USDT")
        
        required_keys = {
            'symbol', 'timestamp', 'overall_sentiment', 'confidence',
            'weighted_score', 'source_stats', 'sample_size'
        }
        assert all(key in result for key in required_keys)
        assert result['symbol'] == "BTC/USDT"
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        
    def test_source_management(self, analyzer):
        """Test source management"""
        # Test initial state
        assert all(analyzer.sources.values())
        
        # Test disabling a source
        analyzer.toggle_source('twitter', False)
        assert not analyzer.sources['twitter']
        assert analyzer.sources['reddit']
        assert analyzer.sources['news']
        
        # Test re-enabling a source
        analyzer.toggle_source('twitter', True)
        assert analyzer.sources['twitter']
        
        # Test invalid source
        with pytest.raises(ValueError):
            analyzer.toggle_source('invalid_source')
