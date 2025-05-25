"""
Unit tests for price predictor
Version 1.0.0 - Created: 2025-05-19 03:44:34 by Patmoorea
"""

import pytest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from src.ai.price_predictor import PricePredictor

@pytest.fixture(autouse=True)
def cleanup_models():
    """Clean up model files after each test"""
    if os.path.exists('models'):
        shutil.rmtree('models')
    os.makedirs('models', exist_ok=True)
    
    yield
    
    if os.path.exists('models'):
        shutil.rmtree('models')

@pytest.fixture
def sample_data():
    """Create sample price data"""
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=500, freq='1h')
    
    # Generate more realistic price movements
    base_price = 30000
    volatility = 100
    returns = np.random.normal(0, volatility, 500)
    price_path = np.exp(np.cumsum(returns/10000))
    prices = base_price * price_path
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, 500)),
        'high': prices * (1 + abs(np.random.normal(0, 0.002, 500))),
        'low': prices * (1 - abs(np.random.normal(0, 0.002, 500))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 500) * price_path
    })
    
    return df.set_index('timestamp')

class TestPricePredictor:
    def test_initialization(self):
        """Test predictor initialization"""
        predictor = PricePredictor('BTC/USDT', '1h')
        assert predictor.symbol == 'BTC/USDT'
        assert predictor.timeframe == '1h'
        assert predictor.sequence_length == 60
        assert predictor.features == ['close', 'volume', 'high', 'low']
        assert os.path.basename(predictor.model_path) == 'BTC_USDT_1h_model'

    def test_data_preparation(self, sample_data):
        """Test data preparation"""
        predictor = PricePredictor('BTC/USDT', '1h')
        X, y = predictor.prepare_data(sample_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1] == predictor.sequence_length
        assert X.shape[2] == len(predictor.features)
        assert len(y) == len(X)

    def test_model_creation(self):
        """Test model creation"""
        predictor = PricePredictor('BTC/USDT', '1h')
        model = predictor._create_model()
        
        assert model is not None
        expected_input_shape = (None, predictor.sequence_length, len(predictor.features))
        assert model.input_shape == expected_input_shape

    def test_training(self, sample_data):
        """Test model training"""
        predictor = PricePredictor('BTC/USDT', '1h')
        history = predictor.train(sample_data, epochs=2, batch_size=32)
        
        assert 'loss' in history
        assert len(history['loss']) > 0
        assert all(isinstance(loss, float) for loss in history['loss'])

    def test_prediction(self, sample_data):
        """Test price prediction"""
        predictor = PricePredictor('BTC/USDT', '1h')
        predictor.train(sample_data, epochs=2, batch_size=32)
        
        result = predictor.predict(sample_data)
        
        assert isinstance(result, dict)
        assert 'predictions' in result
        assert 'confidence' in result
        assert 'timestamp' in result
        assert isinstance(result['predictions'], list)
        assert 0 <= result['confidence'] <= 1
        
        # Test prediction is within reasonable range
        last_price = sample_data['close'].iloc[-1]
        prediction = result['predictions'][0]
        relative_diff = abs(prediction - last_price) / last_price
        assert relative_diff < 0.1, f"Prediction {prediction} too far from last price {last_price}"

    def test_different_timeframes(self):
        """Test with different timeframes"""
        timeframes = ['1h', '4h', '1d']
        for tf in timeframes:
            predictor = PricePredictor('BTC/USDT', tf)
            assert predictor.timeframe == tf
            assert os.path.basename(predictor.model_path) == f'BTC_USDT_{tf}_model'

    def test_sequence_generation(self, sample_data):
        """Test sequence generation"""
        predictor = PricePredictor('BTC/USDT', '1h')
        X, y = predictor.prepare_data(sample_data)
        
        # Check sequence length
        assert X.shape[1] == predictor.sequence_length
        
        # Verify that scalers work correctly
        min_val = predictor.feature_scaler.data_min_
        max_val = predictor.feature_scaler.data_max_
        assert np.all(X >= 0)  # MinMaxScaler output should be >= 0
        assert np.all(X <= 1)  # MinMaxScaler output should be <= 1
