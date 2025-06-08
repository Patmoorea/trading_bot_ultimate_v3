from datetime import datetime

import requests


class GlassnodeClient:
    BASE_URL = "https://api.glassnode.com/v1/metrics/"

    def __init__(self, api_key):
        self.api_key = api_key

    def get_whale_activity(self, asset="btc"):
        endpoint = f"distribution/transfers_volume_sum"
        params = {
            "a": asset,
            "api_key": self.api_key,
            "threshold": 1000000}  # 1M USD
        response = requests.get(self.BASE_URL + endpoint, params=params)
        return response.json()
