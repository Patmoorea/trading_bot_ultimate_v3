import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TimeframeConfig:
    timeframes: List[str] = ("1m", "5m", "15m", "1h", "4h", "1d")
    weights: Dict[str, float] = {
        "1m": 0.1, "5m": 0.15, "15m": 0.2,
        "1h": 0.25, "4h": 0.15, "1d": 0.15
    }

class MultiTimeframeAnalyzer:
    def __init__(self, config: TimeframeConfig = TimeframeConfig()):
        self.config = config
        self.indicators = self._init_indicators()
        
    def _init_indicators(self) -> Dict:
        """Initialise les 42 indicateurs"""
        return {
            "trend": {
                "supertrend": self._calculate_supertrend,
                "ichimoku": self._calculate_ichimoku,
                "vwma": self._calculate_vwma,
                "ema_ribbon": self._calculate_ema_ribbon,
                "parabolic_sar": self._calculate_psar,
                "zigzag": self._calculate_zigzag
            },
            "momentum": {
                "rsi": self._calculate_rsi,
                "stoch_rsi": self._calculate_stoch_rsi,
                "macd": self._calculate_macd,
                "awesome": self._calculate_ao,
                "momentum": self._calculate_momentum,
                "tsi": self._calculate_tsi
            },
            "volatility": {
                "bbands": self._calculate_bbands,
                "keltner": self._calculate_keltner,
                "atr": self._calculate_atr,
                "vix_fix": self._calculate_vix_fix,
                "natr": self._calculate_natr,
                "true_range": self._calculate_tr
            },
            "volume": {
                "obv": self._calculate_obv,
                "vwap": self._calculate_vwap,
                "acc_dist": self._calculate_ad,
                "chaikin_money": self._calculate_cmf,
                "ease_move": self._calculate_emv,
                "volume_profile": self._calculate_vp
            },
            "orderflow": {
                "delta": self._calculate_delta,
                "cvd": self._calculate_cvd,
                "footprint": self._calculate_footprint,
                "liquidity": self._calculate_liquidity,
                "imbalance": self._calculate_imbalance,
                "absorption": self._calculate_absorption
            }
        }
        
    def analyze_timeframe(self, data: pd.DataFrame, 
                         timeframe: str) -> Dict[str, float]:
        """Analyse un timeframe spécifique"""
        results = {}
        for category, inds in self.indicators.items():
            for name, func in inds.items():
                try:
                    results[f"{timeframe}_{name}"] = func(data)
                except Exception as e:
                    print(f"Erreur {name}: {str(e)}")
                    results[f"{timeframe}_{name}"] = None
        return results
        
    def merge_timeframes(self, analyses: Dict[str, Dict]) -> Dict:
        """Fusionne les analyses de tous les timeframes"""
        merged = {}
        for tf, weight in self.config.weights.items():
            if tf in analyses:
                for k, v in analyses[tf].items():
                    if v is not None:
                        key = k.split(f"{tf}_")[1]
                        if key not in merged:
                            merged[key] = 0
                        merged[key] += v * weight
        return merged
