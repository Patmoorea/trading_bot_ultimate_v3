import numpy as np
import talib
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

@dataclass
class IndicatorResult:
    value: float
    signal: str
    confidence: float
    timeframe: str

class TechnicalAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = {
            'trend': ['ichimoku', 'supertrend', 'vwma'],
            'momentum': ['rsi', 'stoch_rsi', 'macd'],
            'volatility': ['atr', 'bb_width', 'keltner'],
            'volume': ['obv', 'vwap', 'accumulation'],
            'orderflow': ['bid_ask_ratio', 'liquidity_wave', 'smart_money_index']
        }
        
    def calculate_all_indicators(self, data: pd.DataFrame, timeframe: str) -> Dict[str, IndicatorResult]:
        results = {}
        try:
            # Trend Indicators
            results.update(self.calculate_trend_indicators(data, timeframe))
            # Momentum Indicators
            results.update(self.calculate_momentum_indicators(data, timeframe))
            # Volatility Indicators
            results.update(self.calculate_volatility_indicators(data, timeframe))
            # Volume Indicators
            results.update(self.calculate_volume_indicators(data, timeframe))
            # OrderFlow Indicators
            results.update(self.calculate_orderflow_indicators(data, timeframe))
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            
        return results

    def calculate_ichimoku(self, data: pd.DataFrame) -> IndicatorResult:
        high = data['high'].values
        low = data['low'].values
        
        tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2
        kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2
        
        if tenkan > kijun:
            signal = "bullish"
            confidence = min((tenkan - kijun) / kijun * 100, 100)
        else:
            signal = "bearish"
            confidence = min((kijun - tenkan) / kijun * 100, 100)
            
        return IndicatorResult(
            value=tenkan,
            signal=signal,
            confidence=confidence,
            timeframe=data.timeframe
        )

    def calculate_supertrend(self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> IndicatorResult:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Calculate SuperTrend
        upper_band = (high + low) / 2 + multiplier * atr
        lower_band = (high + low) / 2 - multiplier * atr
        
        supertrend = np.zeros_like(close)
        direction = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if close[i] > upper_band[i-1]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            elif close[i] < lower_band[i-1]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
                
        current_direction = direction[-1]
        confidence = abs(close[-1] - supertrend[-1]) / close[-1] * 100
        
        return IndicatorResult(
            value=supertrend[-1],
            signal="bullish" if current_direction == 1 else "bearish",
            confidence=min(confidence, 100),
            timeframe=data.timeframe
        )

    def smart_money_index(self, data: pd.DataFrame) -> IndicatorResult:
        volume = data['volume'].values
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Calculate price movement
        price_move = close - np.roll(close, 1)
        
        # Calculate volume weighted price
        volume_price = price_move * volume
        
        # Calculate smart money flow
        smart_money = np.zeros_like(close)
        for i in range(1, len(close)):
            if volume[i] > np.mean(volume[max(0,i-20):i]):
                if close[i] > high[i-1]:
                    smart_money[i] = volume[i]
                elif close[i] < low[i-1]:
                    smart_money[i] = -volume[i]
                    
        smi_value = np.sum(smart_money[-20:])
        confidence = min(abs(smi_value) / np.mean(volume[-20:]) * 100, 100)
        
        return IndicatorResult(
            value=smi_value,
            signal="bullish" if smi_value > 0 else "bearish",
            confidence=confidence,
            timeframe=data.timeframe
        )

    def calculate_combined_signal(self, results: Dict[str, IndicatorResult]) -> Dict:
        trend_score = 0
        momentum_score = 0
        volatility_score = 0
        volume_score = 0
        orderflow_score = 0
        
        for indicator, result in results.items():
            if indicator in self.indicators['trend']:
                trend_score += 1 if result.signal == "bullish" else -1
            elif indicator in self.indicators['momentum']:
                momentum_score += result.confidence/100 if result.signal == "bullish" else -result.confidence/100
            # ... etc pour les autres catégories
            
        return {
            "overall_signal": "bullish" if (trend_score + momentum_score + volume_score + orderflow_score) > 0 else "bearish",
            "confidence": {
                "trend": abs(trend_score) / len(self.indicators['trend']) * 100,
                "momentum": abs(momentum_score) / len(self.indicators['momentum']) * 100,
                "volatility": volatility_score,
                "volume": abs(volume_score) / len(self.indicators['volume']) * 100,
                "orderflow": abs(orderflow_score) / len(self.indicators['orderflow']) * 100
            }
        }
