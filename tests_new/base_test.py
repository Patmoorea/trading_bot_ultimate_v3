import pytest
from config import Config
from decimal import Decimal

class BaseTest:
    @pytest.fixture
    def config(self):
        return Config()
        
    @pytest.fixture
    def base_parameters(self):
        return Config.get_risk_params()

# Add timestamp and user support - Added 2025-05-26 05:50:35 by Patmoorea
def __init__(self):
    """Initialize test with current timestamp and user"""
    self.timestamp = "2025-05-26 05:50:35"  # Current timestamp
    self.user = "Patmoorea"  # Current user
        
def assert_timestamp_format(self, timestamp_str: str) -> bool:
    """Verify timestamp format"""
    pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'
    return bool(re.match(pattern, timestamp_str))

# Add get_test_data method - Added 2025-05-26 05:57:04 by Patmoorea
def get_test_data(self, symbol: str = "BTC/USDT"):
    """Get test data for unit tests"""
    import pandas as pd
    import numpy as np
    
    # Create sample test data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
    data = {
        'timestamp': dates,
        'open': np.random.uniform(30000, 35000, 100),
        'high': np.random.uniform(31000, 36000, 100),
        'low': np.random.uniform(29000, 34000, 100),
        'close': np.random.uniform(30000, 35000, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }
    return pd.DataFrame(data)

    # Update timestamp and user - Added 2025-05-26 06:10:38 by Patmoorea
    def __init__(self):
        super().__init__()
        self.timestamp = "2025-05-26 06:10:38"
        self.user = "Patmoorea"

# Add test data decorator - Added 2025-05-26 06:10:38 by Patmoorea
def with_test_data(cls):
    """Class decorator to add get_test_data method"""
    def get_test_data(self, *args, **kwargs):
        """Generate test data for tests"""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(30000, 35000, 100),
            'high': np.random.uniform(31000, 36000, 100),
            'low': np.random.uniform(29000, 34000, 100),
            'close': np.random.uniform(30000, 35000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }
        return pd.DataFrame(data)
    
    cls.get_test_data = get_test_data
    return cls

    # Update timestamp and user - Added 2025-05-26 06:14:57 by Patmoorea
    def __init__(self):
        self.timestamp = "2025-05-26 06:14:57"
        self.user = "Patmoorea"

    # Update timestamp - Added 2025-05-26 14:31:40 by Patmoorea
    def get_timestamp(self):
        """Get current timestamp"""
        return "2025-05-26 14:31:40"
        
    def get_user(self):
        """Get current user"""
        return "Patmoorea"

    # Add get_test_data method - Added 2025-05-26 14:31:40 by Patmoorea
    def get_test_data(self, symbol="BTC/USDT", *args, **kwargs):
        """Get test data for unit tests
        
        Args:
            symbol (str): Trading pair symbol
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample test data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='H')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(30000, 35000, 100),
            'high': np.random.uniform(31000, 36000, 100),
            'low': np.random.uniform(29000, 34000, 100),
            'close': np.random.uniform(30000, 35000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }
        return pd.DataFrame(data)

# Update timestamp and user - Added 2025-05-26 14:40:58 by Patmoorea
def setUp(self):
    """Setup test data with correct timestamp"""
    self.timestamp = "2025-05-26 14:40:58"
    self.user = "Patmoorea"

# Fix get_test_data signature - Added 2025-05-26 14:40:58 by Patmoorea
def get_test_data(self, *args, **kwargs):
    """Get test data for unit tests with flexible arguments
    
    Args:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
        
    Returns:
        DataFrame: Test data frame with OHLCV data
    """
    import pandas as pd
    import numpy as np
    
    symbol = kwargs.get('symbol', 'BTC/USDT')
    periods = kwargs.get('periods', 100)
    
    dates = pd.date_range(start='2025-01-01', periods=periods, freq='H')
    data = {
        'timestamp': dates,
        'open': np.random.uniform(30000, 35000, periods),
        'high': np.random.uniform(31000, 36000, periods),
        'low': np.random.uniform(29000, 34000, periods),
        'close': np.random.uniform(30000, 35000, periods),
        'volume': np.random.uniform(100, 1000, periods)
    }
    return pd.DataFrame(data)
