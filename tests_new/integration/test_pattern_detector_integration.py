"""
Integration tests for pattern detector
Version 1.0.0 - Created: 2025-05-19 02:41:58 by Patmoorea
"""

import pytest
import pandas as pd
import numpy as np
from src.ai.pattern_detector import PatternDetector, Pattern
import pytz
from datetime import datetime

@pytest.fixture(scope="module")
def sample_data():
    """Create sample price data with known patterns"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1h')
    
    # Create a double top pattern
    high_values = []
    for i in range(100):
        if i in [30, 50]:  # Create two peaks
            high_values.append(35000)
        else:
            high_values.append(np.random.uniform(31000, 34000))
    
    volumes = np.random.uniform(100, 1000, 100)
    volumes[50] = 2000  # High volume at second peak
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(30000, 35000, 100),
        'high': high_values,
        'low': np.random.uniform(29000, 34000, 100),
        'close': np.random.uniform(30000, 35000, 100),
        'volume': volumes
    })
    return df.set_index('timestamp')

@pytest.fixture(scope="module")
def detector():
    return PatternDetector()

class TestPatternDetectorIntegration:
    def test_pattern_detection_pipeline(self, detector, sample_data):
        """Test complete pattern detection pipeline"""
        # Detect all patterns
        patterns = detector.detect_all(sample_data)
        
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, dict)
            assert all(k in pattern for k in ['pattern', 'timestamp', 'confidence', 'price_level'])
            
            # Test volume confirmation
            has_volume = detector.confirm_volume(sample_data, pattern)
            assert isinstance(has_volume, bool)
    
    def test_pattern_consistency(self, detector, sample_data):
        """Test consistency of pattern detection"""
        patterns1 = detector.detect_all(sample_data)
        patterns2 = detector.detect_all(sample_data)
        
        # Results should be consistent
        assert len(patterns1) == len(patterns2)
        
        if patterns1:
            # Compare key attributes
            assert patterns1[0]['pattern'] == patterns2[0]['pattern']
            assert patterns1[0]['price_level'] == patterns2[0]['price_level']
            assert patterns1[0]['confidence'] == patterns2[0]['confidence']
    
    def test_volume_analysis(self, detector, sample_data):
        """Test volume analysis with pattern detection"""
        patterns = detector.detect_all(sample_data)
        
        if patterns:
            pattern = patterns[0]
            # Create periods of high and low volume
            sample_data.loc[sample_data.index[-10:], 'volume'] *= 2
            
            # Test volume confirmation
            high_volume = detector.confirm_volume(sample_data, pattern)
            assert isinstance(high_volume, bool)
            assert high_volume is True
            
            # Reset volumes to low
            sample_data.loc[sample_data.index[-10:], 'volume'] /= 4
            
            # Test volume rejection
            low_volume = detector.confirm_volume(sample_data, pattern)
            assert isinstance(low_volume, bool)
            assert low_volume is False
            
    def test_multiple_timeframes(self, detector):
        """Test pattern detection across multiple timeframes"""
        timeframes = ['1h', '4h', '1d']
        
        for tf in timeframes:
            periods = 100 if tf == '1h' else 50
            df = pd.DataFrame({
                'timestamp': pd.date_range(start='2025-01-01', periods=periods, freq=tf),
                'open': np.random.uniform(30000, 35000, periods),
                'high': np.random.uniform(31000, 36000, periods),
                'low': np.random.uniform(29000, 34000, periods),
                'close': np.random.uniform(30000, 35000, periods),
                'volume': np.random.uniform(100, 1000, periods)
            }).set_index('timestamp')
            
            patterns = detector.detect_all(df)
            assert isinstance(patterns, list)

    def test_error_handling(self, detector, sample_data):
        """Test error handling in integration"""
        # Test with corrupted data
        corrupted_data = sample_data.copy()
        corrupted_data.loc[corrupted_data.index[-1], 'high'] = None
        
        patterns = detector.detect_all(corrupted_data)
        assert isinstance(patterns, list)
        
        # Test with missing columns
        missing_columns = sample_data.drop(['volume'], axis=1)
        with pytest.raises(ValueError):
            detector.detect_all(missing_columns)
