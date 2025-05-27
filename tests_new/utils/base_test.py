import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class BaseTestClass:
    @staticmethod
    def get_test_data(*args, **kwargs):
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'price': np.random.uniform(30000, 35000, 100),
            'volume': np.random.uniform(1, 10, 100),
            'close': np.random.uniform(30000, 35000, 100),
            'open': np.random.uniform(30000, 35000, 100),
            'high': np.random.uniform(31000, 36000, 100),
            'low': np.random.uniform(29000, 34000, 100)
        })

# Tous les tests techniques héritent de BaseTestClass
class TestAIFull(BaseTestClass):
    pass

class TestHybridAI(BaseTestClass):
    pass

class TestTrain(BaseTestClass):
    pass

class TestIndicatorsV2(BaseTestClass):
    pass

class TestTechnical(BaseTestClass):
    pass

class TestRSIMinimal(BaseTestClass):
    pass

class TestTechnicalMinimal(BaseTestClass):
    pass

class TestDataV2(BaseTestClass):
    pass

class TestPerformance(BaseTestClass):
    pass
