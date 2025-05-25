import pytest
from src.ai.sentiment_analyzer import SentimentAnalyzer

@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()

class TestSentimentAnalyzerIntegration:
    def test_full_analysis_cycle(self, analyzer):
        """Test complete sentiment analysis cycle"""
        symbol = "BTC/USDT"
        result = analyzer.get_aggregated_sentiment(symbol)
        
        assert result['symbol'] == symbol
        assert isinstance(result['confidence'], float)
        assert all(k in result['source_stats'] for k in ['twitter', 'reddit', 'news'])
        
    def test_source_effects(self, analyzer):
        """Test effects of enabling/disabling sources"""
        # Get base result
        base_result = analyzer.get_aggregated_sentiment("BTC/USDT")
        base_size = base_result['sample_size']
        
        # Disable one source
        analyzer.toggle_source('twitter', False)
        partial_result = analyzer.get_aggregated_sentiment("BTC/USDT")
        partial_size = partial_result['sample_size']
        
        # Disable another source
        analyzer.toggle_source('reddit', False)
        minimal_result = analyzer.get_aggregated_sentiment("BTC/USDT")
        minimal_size = minimal_result['sample_size']
        
        # Verify sample sizes decrease as expected
        assert base_size > partial_size > minimal_size
        
        # Restore original state
        analyzer.toggle_source('twitter', True)
        analyzer.toggle_source('reddit', True)
        
    def test_consistency(self, analyzer):
        """Test consistency of results"""
        symbol = "BTC/USDT"
        results = [analyzer.get_aggregated_sentiment(symbol) for _ in range(3)]
        
        # Verify all results have the same structure
        assert all(set(r.keys()) == set(results[0].keys()) for r in results)
        
        # Verify source stats are consistent
        assert all(r['source_stats'].keys() == results[0]['source_stats'].keys() 
                  for r in results)
