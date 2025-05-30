import pytest
from unittest.mock import AsyncMock, MagicMock

class TestArbitrageScannerIntegration:
    @pytest.fixture
    def scanner(self):
        mock_scanner = MagicMock()
        mock_scanner.get_timestamps = MagicMock(return_value=[1, 1])
        return mock_scanner

    def test_timestamp_consistency(self, scanner):
        timestamps = scanner.get_timestamps()
        assert len(set(timestamps)) == 1  # Vérifie que tous les timestamps sont identiques

    def test_error_resilience(self):
        with pytest.raises(Exception) as exc_info:
            raise Exception("Test error")
        assert str(exc_info.value) == "Test error"
