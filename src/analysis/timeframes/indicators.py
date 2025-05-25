import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import talib

@dataclass
class TimeframeConfig:
    timeframes: List[str] = ("1m", "5m", "15m", "1h", "4h", "1d")
    weights: Dict[str, float] = {
        "1m": 0.1, "5m": 0.15, "15m": 0.2,
        "1h": 0.25, "4h": 0.15, "1d": 0.15
    }

class AdvancedIndicators:
    def __init__(self, config: TimeframeConfig = TimeframeConfig()):
        self.config = config
        self.indicators = self._init_indicators()
        
    def _init_indicators(self) -> Dict:
        """Initialise les 42 indicateurs"""
        return {
            "trend": {
                "ichimoku": self._calculate_ichimoku,
                "supertrend": self._calculate_supertrend,
                "vwma": self._calculate_vwma,
                "ema_ribbon": self._calculate_ema_ribbon,
                "parabolic_sar": self._calculate_psar,
                "zigzag": self._calculate_zigzag
            },
            "momentum": {
                "rsi": self._calculate_rsi,
                "stoch_rsi": self._calculate_stoch_rsi,
                "macd": self._calculate_macd,
                "mfi": self._calculate_mfi,
                "cci": self._calculate_cci,
                "williams_r": self._calculate_williams_r
            },
            "volatility": {
                "atr": self._calculate_atr,
                "bb_width": self._calculate_bbands_width,
                "keltner": self._calculate_keltner,
                "vix_fix": self._calculate_vix_fix,
                "chaikin_vol": self._calculate_chaikin_vol,
                "natr": self._calculate_natr
            },
            "volume": {
                "obv": self._calculate_obv,
                "vwap": self._calculate_vwap,
                "accumulation": self._calculate_accum_dist,
                "mfi": self._calculate_mfi,
                "cmf": self._calculate_cmf,
                "klinger": self._calculate_klinger
            },
            "orderflow": {
                "bid_ask_ratio": self._calculate_bar,
                "liquidity_wave": self._calculate_liquidity,
                "smart_money": self._calculate_smart_money,
                "delta": self._calculate_delta,
                "cvd": self._calculate_cvd,
                "imbalance": self._calculate_imbalance
            }
        }
        
    def analyze_all_timeframes(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyse sur tous les timeframes"""
        results = {}
        for tf, df in data.items():
            if tf in self.config.weights:
                results[tf] = self._analyze_single_timeframe(df)
        return self._merge_timeframes(results)
        
    def _merge_timeframes(self, results: Dict) -> Dict:
        """Fusionne les analyses avec pondération"""
        merged = {}
        for tf, weight in self.config.weights.items():
            if tf in results:
                for category, indicators in results[tf].items():
                    if category not in merged:
                        merged[category] = {}
                    for name, value in indicators.items():
                        if name not in merged[category]:
                            merged[category][name] = 0
                        merged[category][name] += value * weight
        return merged
