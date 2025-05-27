import pytest
from datetime import datetime

class TestCore:
    @pytest.fixture
    def core_config(self):
        return {
            'timestamp': datetime(2025, 5, 27, 7, 25, 20),
            'user': 'Patmoorea',
            'environment': {
                'python': '3.11.9',
                'pytest': '8.0.0',
                'os': 'macOS 15.3.2',
                'arch': 'arm64'
            },
            'hardware': {
                'cpu': 'Apple M4',
                'ram': '16 Go',
                'storage': '460Gi'
            }
        }

    def test_core_configuration(self, core_config):
        assert isinstance(core_config['timestamp'], datetime)
        assert core_config['user'] == 'Patmoorea'
        assert core_config['environment']['python'].startswith('3.11')
        assert core_config['hardware']['cpu'] == 'Apple M4'

    @pytest.fixture
    def performance_settings(self):
        return {
            'max_threads': 10,
            'max_memory': '12 Go',
            'storage_limit': '100Gi'
        }

    def test_performance_limits(self, performance_settings):
        assert performance_settings['max_threads'] == 10
        assert performance_settings['max_memory'] == '12 Go'
        assert performance_settings['storage_limit'] == '100Gi'
