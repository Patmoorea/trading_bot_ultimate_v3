"""
Configuration des exchanges - Mise à jour: 2025-05-17 23:03:03
@author: Patmoorea
"""
class ExchangesConfig:
    SUPPORTED_EXCHANGES = {
        'binance': {
            'ws_url': 'wss://stream.binance.com:9443/ws',
            'rest_url': 'https://api.binance.com/api/v3',
            'test_url': 'https://testnet.binance.vision/api'
        },
        'gateio': {
            'ws_url': 'wss://ws.gate.io/v4',
            'rest_url': 'https://api.gateio.ws/api/v4',
            'test_url': 'https://fx-api-testnet.gateio.ws/api/v4'
        },
        'bingx': {
            'ws_url': 'wss://open-api-swap.bingx.com/swap-market',
            'rest_url': 'https://open-api.bingx.com/api/v1',
            'test_url': 'https://open-api-swap.bingx.com/testnet'
        },
        'okx': {
            'ws_url': 'wss://ws.okx.com:8443/ws/v5',
            'rest_url': 'https://www.okx.com/api/v5',
            'test_url': 'https://www.okx.com/api-test/v5'
        },
        'blofin': {
            'ws_url': 'wss://ws.blofin.com/api/ws',
            'rest_url': 'https://api.blofin.com/api/v1',
            'test_url': 'https://api-testnet.blofin.com/api/v1'
        }
    }
    
    @classmethod
    def get_exchange_config(cls, exchange_name: str) -> dict:
        return cls.SUPPORTED_EXCHANGES.get(exchange_name, {})
