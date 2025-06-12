import pytest
from datetime import datetime

class TestBase:
    @pytest.fixture
    def system_info(self):
        return {
            'timestamp': datetime(2025, 5, 27, 7, 25, 20),
            'user': 'Patmoorea',
            'hardware': {
                'cpu': 'Apple M4',
                'cores': 10,
                'ram': '16 Go',
                'gpu': 'Apple M4',
                'storage': {
                    'total': '460Gi',
                    'free': '113Gi'
                }
            },
            'software': {
                'os': 'macOS 15.3.2',
                'kernel': 'Darwin 24.3.0 arm64'
            }
        }

    def test_system_configuration(self, system_info):
        assert isinstance(system_info['timestamp'], datetime)
        assert system_info['user'] == 'Patmoorea'
        assert system_info['hardware']['cpu'] == 'Apple M4'
        assert system_info['hardware']['cores'] == 10
        assert system_info['hardware']['ram'] == '16 Go'
        assert system_info['software']['os'] == 'macOS 15.3.2'

    @pytest.fixture
    def test_config(self):
        return {
            'debug': True,
            'async_mode': True,
            'test_date': datetime(2025, 5, 27, 7, 25, 20)
        }

    def test_configuration(self, test_config):
        assert isinstance(test_config['test_date'], datetime)
        assert test_config['debug'] is True
        assert test_config['async_mode'] is True
