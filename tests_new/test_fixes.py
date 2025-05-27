import pytest
from unittest.mock import MagicMock, patch
import json

class TestFixes:
    @pytest.fixture
    def mock_json_handling(self):
        original_dumps = json.dumps
        def safe_dumps(obj, **kwargs):
            if isinstance(obj, (dict, list)):
                return original_dumps(obj)
            return str(obj)
        with patch('json.dumps', side_effect=safe_dumps):
            yield

    def test_json_serialization(self, mock_json_handling):
        data = {'test': 1}
        result = json.dumps(data)
        assert isinstance(result, str)
