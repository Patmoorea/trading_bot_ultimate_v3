import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class TimeframeConfig:
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    weights: Dict[str, float] = field(default_factory=lambda: {
        "1m": 0.1, "5m": 0.15, "15m": 0.2,
        "1h": 0.25, "4h": 0.15, "1d": 0.15
    })

class MultiTimeframeAnalyzer:
    def __init__(self, config: TimeframeConfig = TimeframeConfig()):
        self.config = config
        self.indicators = self._init_indicators()
        
    def _init_indicators(self) -> Dict:
        """Initialise 45+ indicateurs techniques"""
        return {
            "trend": {
                "supertrend": self._calculate_supertrend,
                "ichimoku": self._calculate_ichimoku,
                "vwma": self._calculate_vwma,
                "ema_ribbon": self._calculate_ema_ribbon,
                "parabolic_sar": self._calculate_psar,
                "zigzag": self._calculate_zigzag,
                "dema": self._calculate_dema,
                "tema": self._calculate_tema,
                "trix": self._calculate_trix,
                "adx": self._calculate_adx,
                "di_plus": self._calculate_di_plus,
                "di_minus": self._calculate_di_minus
            },
            "momentum": {
                "rsi": self._calculate_rsi,
                "stoch_rsi": self._calculate_stoch_rsi,
                "macd": self._calculate_macd,
                "awesome": self._calculate_ao,
                "momentum": self._calculate_momentum,
                "tsi": self._calculate_tsi,
                "cci": self._calculate_cci,
                "williams_r": self._calculate_williams_r,
                "dpo": self._calculate_dpo,
                "kst": self._calculate_kst,
                "ppo": self._calculate_ppo,
                "roc": self._calculate_roc
            },
            "volatility": {
                "bbands": self._calculate_bbands,
                "keltner": self._calculate_keltner,
                "atr": self._calculate_atr,
                "vix_fix": self._calculate_vix_fix,
                "natr": self._calculate_natr,
                "true_range": self._calculate_tr,
                "standard_dev": self._calculate_std_dev,
                "donchian": self._calculate_donchian,
                "chandelier": self._calculate_chandelier,
                "chaos_bands": self._calculate_chaos_bands,
                "rv": self._calculate_realized_volatility,
                "atr_bands": self._calculate_atr_bands
            },
            "volume": {
                "obv": self._calculate_obv,
                "vwap": self._calculate_vwap,
                "acc_dist": self._calculate_ad,
                "chaikin_money": self._calculate_cmf,
                "ease_move": self._calculate_emv,
                "volume_profile": self._calculate_vp,
                "mfi": self._calculate_mfi,
                "vpt": self._calculate_vpt,
                "nvi": self._calculate_nvi,
                "pvi": self._calculate_pvi,
                "vwmacd": self._calculate_vwmacd,
                "klinger": self._calculate_klinger
            },
            "orderflow": {
                "delta": self._calculate_delta,
                "cvd": self._calculate_cvd,
                "footprint": self._calculate_footprint,
                "liquidity": self._calculate_liquidity,
                "imbalance": self._calculate_imbalance,
                "absorption": self._calculate_absorption,
                "vwap_zones": self._calculate_vwap_zones,
                "poc_analysis": self._calculate_poc,
                "order_blocks": self._calculate_order_blocks,
                "smart_money": self._calculate_smart_money,
                "whale_alerts": self._calculate_whale_alerts,
                "funding_rate": self._calculate_funding_rate
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
