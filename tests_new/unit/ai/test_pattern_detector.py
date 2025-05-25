"""
Unit tests for pattern detector
Version 1.0.0 - Created: 2025-05-19 02:41:58 by Patmoorea
"""

import pytest
import pandas as pd
import numpy as np
from src.ai.pattern_detector import PatternDetector, Pattern
from datetime import datetime
import pytz

@pytest.fixture
def sample_data():
    """Create sample price data"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(30000, 35000, 100),
        'high': np.random.uniform(31000, 36000, 100),
        'low': np.random.uniform(29000, 34000, 100),
        'close': np.random.uniform(30000, 35000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    })
    return df.set_index('timestamp')

@pytest.fixture
def detector():
    return PatternDetector()

@pytest.fixture
def pattern():
    return {
        'pattern': Pattern.BREAKOUT,
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'confidence': 0.8,
        'price_level': 35000
    }

class TestPatternDetector:
    def test_initialization(self, detector):
        """Test detector initialization"""
        assert isinstance(detector.patterns, dict)
        assert len(detector.patterns) > 0
        assert all(callable(func) for func in detector.patterns.values())
        assert detector.volume_threshold == 1.5

    def test_data_validation(self, detector, sample_data):
        """Test data validation"""
        assert detector._validate_data(sample_data)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        assert not detector._validate_data(invalid_data)

    def test_pattern_detection(self, detector, sample_data):
        """Test pattern detection"""
        patterns = detector.detect_all(sample_data)
        assert isinstance(patterns, list)
        
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, dict)
            assert 'pattern' in pattern
            assert 'timestamp' in pattern
            assert 'confidence' in pattern
            assert 'price_level' in pattern

    def test_volume_confirmation(self, detector, sample_data, pattern):
        """Test volume confirmation"""
        # Create high volume condition
        sample_data.loc[sample_data.index[-1], 'volume'] = 2000  # High volume
        high_volume = detector.confirm_volume(sample_data, pattern)
        assert isinstance(high_volume, bool)
        assert high_volume is True

        # Create low volume condition
        sample_data.loc[sample_data.index[-1], 'volume'] = 50  # Low volume
        low_volume = detector.confirm_volume(sample_data, pattern)
        assert isinstance(low_volume, bool)
        assert low_volume is False

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        assert not detector.confirm_volume(empty_df, pattern)

        # Test with missing volume column
        no_volume_df = sample_data.drop('volume', axis=1)
        assert not detector.confirm_volume(no_volume_df, pattern)

    def test_double_top(self, detector, sample_data):
        """Test double top detection"""
        # Create double top pattern in data
        sample_data.loc[sample_data.index[30], 'high'] = 35000
        sample_data.loc[sample_data.index[50], 'high'] = 35000
        
        patterns = detector._detect_double_top(sample_data)
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert pattern['pattern'] == Pattern.DOUBLE_TOP
            assert 0.7 <= pattern['confidence'] <= 0.9

    def test_double_bottom(self, detector, sample_data):
        """Test double bottom detection"""
        # Create double bottom pattern in data
        sample_data.loc[sample_data.index[30], 'low'] = 29000
        sample_data.loc[sample_data.index[50], 'low'] = 29000
        
        patterns = detector._detect_double_bottom(sample_data)
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert pattern['pattern'] == Pattern.DOUBLE_BOTTOM
            assert 0.7 <= pattern['confidence'] <= 0.9

    def test_head_shoulders(self, detector, sample_data):
        """Test head and shoulders detection"""
        # Create head and shoulders pattern in data
        sample_data.loc[sample_data.index[30], 'high'] = 34000  # Left shoulder
        sample_data.loc[sample_data.index[40], 'high'] = 35000  # Head
        sample_data.loc[sample_data.index[50], 'high'] = 34000  # Right shoulder
        
        patterns = detector._detect_head_shoulders(sample_data)
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert pattern['pattern'] == Pattern.HEAD_SHOULDERS
            assert 0.6 <= pattern['confidence'] <= 0.8

    def test_triangle(self, detector, sample_data):
        """Test triangle detection"""
        # Create converging highs and lows
        for i in range(20):
            sample_data.loc[sample_data.index[-20+i], 'high'] = 35000 - (i * 50)
            sample_data.loc[sample_data.index[-20+i], 'low'] = 30000 + (i * 50)
        
        patterns = detector._detect_triangle(sample_data)
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert pattern['pattern'] == Pattern.TRIANGLE
            assert 0.5 <= pattern['confidence'] <= 0.7

    def test_breakout(self, detector, sample_data):
        """Test breakout detection"""
        # Create breakout pattern
        sample_data.loc[sample_data.index[-1], 'close'] = 36000  # Upward breakout
        
        patterns = detector._detect_breakout(sample_data)
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert pattern['pattern'] == Pattern.BREAKOUT
            assert 'direction' in pattern
            assert pattern['direction'] in ['up', 'down']
            assert 0.7 <= pattern['confidence'] <= 0.8

    def test_error_handling(self, detector):
        """Test error handling"""
        # Test with None
        assert not detector.confirm_volume(None, {})
        
        # Test with invalid DataFrame
        assert not detector.confirm_volume(pd.DataFrame(), {})
        
        # Test with invalid pattern
        assert not detector.confirm_volume(pd.DataFrame({'volume': [1, 2, 3]}), {})
