import pytest
from tests_new.base_test import BaseTest

class TestHybrid(BaseTest):
    @pytest.mark.skip(reason="Backward compatibility not implemented yet")
    def test_backward_compatibility(self):
        """Test backward compatibility"""
        pass
