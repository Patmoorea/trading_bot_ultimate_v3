import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import talib
from talib import abstract

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
        """Initialise les indicateurs techniques"""
        return {
            "trend": {
                "sma": self._calculate_sma,
                "ema": self._calculate_ema,
                "macd": self._calculate_macd,
                "adx": self._calculate_adx
            },
            "momentum": {
                "rsi": self._calculate_rsi,
                "stoch": self._calculate_stoch,
                "willr": self._calculate_willr,
                "mom": self._calculate_mom
            },
            "volatility": {
                "bbands": self._calculate_bbands,
                "atr": self._calculate_atr
            },
            "volume": {
                "obv": self._calculate_obv,
                "ad": self._calculate_ad
            }
        }
        
    def _calculate_sma(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calcule la Simple Moving Average"""
        try:
            close = data['close'].values
            sma = talib.SMA(close, timeperiod=period)
            return sma[-1]
        except Exception as e:
            print(f"Erreur SMA: {str(e)}")
            return None
            
    def _calculate_ema(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calcule l'Exponential Moving Average"""
        try:
            close = data['close'].values
            ema = talib.EMA(close, timeperiod=period)
            return ema[-1]
        except Exception as e:
            print(f"Erreur EMA: {str(e)}")
            return None

    def _calculate_macd(self, data: pd.DataFrame) -> dict:
        """Calcule le MACD"""
        try:
            close = data['close'].values
            macd, signal, hist = talib.MACD(close)
            return {
                'macd': macd[-1],
                'signal': signal[-1],
                'hist': hist[-1]
            }
        except Exception as e:
            print(f"Erreur MACD: {str(e)}")
            return None

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcule le RSI"""
        try:
            close = data['close'].values
            rsi = talib.RSI(close, timeperiod=period)
            return rsi[-1]
        except Exception as e:
            print(f"Erreur RSI: {str(e)}")
            return None

    def _calculate_stoch(self, data: pd.DataFrame) -> dict:
        """Calcule le Stochastic"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            slowk, slowd = talib.STOCH(high, low, close)
            return {
                'k': slowk[-1],
                'd': slowd[-1]
            }
        except Exception as e:
            print(f"Erreur Stochastic: {str(e)}")
            return None

    def _calculate_bbands(self, data: pd.DataFrame) -> dict:
        """Calcule les Bollinger Bands"""
        try:
            close = data['close'].values
            upper, middle, lower = talib.BBANDS(close)
            return {
                'upper': upper[-1],
                'middle': middle[-1],
                'lower': lower[-1]
            }
        except Exception as e:
            print(f"Erreur BBands: {str(e)}")
            return None

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcule l'Average True Range"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            atr = talib.ATR(high, low, close, timeperiod=period)
            return atr[-1]
        except Exception as e:
            print(f"Erreur ATR: {str(e)}")
            return None

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcule l'Average Directional Index"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            adx = talib.ADX(high, low, close, timeperiod=period)
            return adx[-1]
        except Exception as e:
            print(f"Erreur ADX: {str(e)}")
            return None

    def _calculate_willr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcule le Williams %R"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            willr = talib.WILLR(high, low, close, timeperiod=period)
            return willr[-1]
        except Exception as e:
            print(f"Erreur WILLR: {str(e)}")
            return None

    def _calculate_mom(self, data: pd.DataFrame, period: int = 10) -> float:
        """Calcule le Momentum"""
        try:
            close = data['close'].values
            mom = talib.MOM(close, timeperiod=period)
            return mom[-1]
        except Exception as e:
            print(f"Erreur Momentum: {str(e)}")
            return None

    def _calculate_obv(self, data: pd.DataFrame) -> float:
        """Calcule l'On Balance Volume"""
        try:
            close = data['close'].values
            volume = data['volume'].values
            obv = talib.OBV(close, volume)
            return obv[-1]
        except Exception as e:
            print(f"Erreur OBV: {str(e)}")
            return None

    def _calculate_ad(self, data: pd.DataFrame) -> float:
        """Calcule l'Accumulation/Distribution"""
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            ad = talib.AD(high, low, close, volume)
            return ad[-1]
        except Exception as e:
            print(f"Erreur A/D: {str(e)}")
            return None

    def analyze_timeframe(self, data: Dict[str, pd.DataFrame], timeframe: str) -> Dict[str, Dict]:
        """Analyse un timeframe spécifique"""
        results = {}
        
        # Pour chaque symbole dans les données
        for symbol, df in data.items():
            results[symbol] = {}
            
            # Pour chaque catégorie d'indicateurs
            for category, indicators in self.indicators.items():
                results[symbol][category] = {}
                
                # Pour chaque indicateur dans la catégorie
                for name, func in indicators.items():
                    try:
                        value = func(df)
                        if isinstance(value, dict):
                            for k, v in value.items():
                                results[symbol][category][f"{name}_{k}"] = v
                        else:
                            results[symbol][category][name] = value
                    except Exception as e:
                        print(f"Erreur calcul {name} pour {symbol}: {str(e)}")
                        results[symbol][category][name] = None

        return results

    def merge_timeframes(self, analyses: Dict[str, Dict]) -> Dict:
        """Fusionne les analyses de tous les timeframes"""
        merged = {}
        for tf, weight in self.config.weights.items():
            if tf in analyses:
                for symbol, symbol_data in analyses[tf].items():
                    if symbol not in merged:
                        merged[symbol] = {}
                    
                    for category, indicators in symbol_data.items():
                        if category not in merged[symbol]:
                            merged[symbol][category] = {}
                            
                        for name, value in indicators.items():
                            if value is not None:
                                if name not in merged[symbol][category]:
                                    merged[symbol][category][name] = 0
                                merged[symbol][category][name] += value * weight
                                
        return merged
