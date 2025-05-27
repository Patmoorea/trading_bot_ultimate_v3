import pytest
from unittest.mock import MagicMock, patch
import sys

def test_modules_mock():
    mock_module = MagicMock()
    mock_module.news = MagicMock()
    with patch.dict(sys.modules, {'modules': mock_module}):
        from modules import news
        assert hasattr(news, 'process')
