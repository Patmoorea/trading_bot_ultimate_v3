from typing import Dict, List, Union
import numpy as np
import pandas as pd
from talib import abstract

class AdvancedIndicators:
    def __init__(self):
        self.indicators = {
            'trend': self._trend_indicators(),
            'momentum': self._momentum_indicators(),
            'volatility': self._volatility_indicators(),
            'volume': self._volume_indicators(),
            'orderflow': self._orderflow_indicators(),
            'custom': self._custom_indicators()
        }
        self.cache = {}

    def _trend_indicators(self) -> Dict:
        return {
            'ichimoku': self._ichimoku,
            'supertrend': self._supertrend,
            'vwma': self._vwma,
            'trix': self._trix,
            'dema': self._dema,
            'tema': self._tema,
            'kama': self._kama
        }

    def _momentum_indicators(self) -> Dict:
        return {
            'rsi': self._rsi,
            'stoch_rsi': self._stoch_rsi,
            'macd': self._macd,
            'cci': self._cci,
            'williams_r': self._williams_r,
            'ultimate_oscillator': self._ultimate_oscillator,
            'ao': self._awesome_oscillator
        }

    def _volatility_indicators(self) -> Dict:
        return {
            'atr': self._atr,
            'bb_width': self._bollinger_bandwidth,
            'keltner': self._keltner_channels,
            'parkinson': self._parkinson_volatility,
            'yang_zhang': self._yang_zhang_volatility,
            'chaikin_volatility': self._chaikin_volatility
        }

    def _volume_indicators(self) -> Dict:
        return {
            'obv': self._obv,
            'vwap': self._vwap,
            'accumulation': self._accumulation_distribution,
            'cmf': self._chaikin_money_flow,
            'mfi': self._money_flow_index,
            'eom': self._ease_of_movement
        }

    def _orderflow_indicators(self) -> Dict:
        return {
            'bid_ask_ratio': self._bid_ask_ratio,
            'liquidity_wave': self._liquidity_wave,
            'smart_money_index': self._smart_money_index,
            'delta_volume': self._delta_volume,
            'imbalance': self._order_imbalance
        }

    def _custom_indicators(self) -> Dict:
        return {
            'custom1': self._custom_indicator1,
            'custom2': self._custom_indicator2
        }

    def calculate_all(self, data: pd.DataFrame) -> Dict:
        results = {}
        for category, indicators in self.indicators.items():
            results[category] = {
                name: func(data) 
                for name, func in indicators.items()
            }
        return results

    def _cache_result(func):
        def wrapper(self, data, *args, **kwargs):
            key = f"{func.__name__}_{hash(str(data.index[-1]))}"
            if key not in self.cache:
                self.cache[key] = func(self, data, *args, **kwargs)
            return self.cache[key]
        return wrapper

    @_cache_result
    def _ichimoku(self, data: pd.DataFrame) -> Dict:
        tenkan = self._calc_midpoint(data, 9)
        kijun = self._calc_midpoint(data, 26)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = self._calc_midpoint(data, 52)
        chikou = data['close'].shift(-26)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }

    def _calc_midpoint(self, data: pd.DataFrame, period: int) -> pd.Series:
        high = data['high'].rolling(period).max()
        low = data['low'].rolling(period).min()
        return (high + low) / 2

    @_cache_result
    def _supertrend(self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict:
        atr = self._atr(data, period)
        hl2 = (data['high'] + data['low']) / 2
        
        upper_basic = hl2 + (multiplier * atr)
        lower_basic = hl2 - (multiplier * atr)
        
        upper_band = pd.Series(index=data.index, dtype='float64')
        lower_band = pd.Series(index=data.index, dtype='float64')
        
        for i in range(len(data)):
            if i == 0:
                upper_band[i] = upper_basic[i]
                lower_band[i] = lower_basic[i]
            else:
                upper_band[i] = upper_basic[i] if (data['close'][i-1] > upper_band[i-1]) else min(upper_basic[i], upper_band[i-1])
                lower_band[i] = lower_basic[i] if (data['close'][i-1] < lower_band[i-1]) else max(lower_basic[i], lower_band[i-1])
        
        return {
            'upper': upper_band,
            'lower': lower_band,
            'trend': pd.Series(np.where(data['close'] > upper_band, 1, -1), index=data.index)
        }

    @_cache_result
    def _vwma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        return (data['close'] * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()

    @_cache_result
    def _trix(self, data: pd.DataFrame, period: int = 18) -> pd.Series:
        ema1 = data['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100

    @_cache_result
    def _dema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        ema = data['close'].ewm(span=period, adjust=False).mean()
        return 2 * ema - ema.ewm(span=period, adjust=False).mean()

    @_cache_result
    def _tema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        ema1 = data['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    @_cache_result
    def _kama(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        change = abs(data['close'] - data['close'].shift(period))
        volatility = abs(data['close'] - data['close'].shift(1)).rolling(period).sum()
        er = change / volatility
        fastest = 2 / (2 + 1)
        slowest = 2 / (30 + 1)
        sc = (er * (fastest - slowest) + slowest) ** 2
        
        kama = pd.Series(index=data.index, dtype='float64')
        kama.iloc[period-1] = data['close'].iloc[period-1]
        for i in range(period, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data['close'].iloc[i] - kama.iloc[i-1])
        return kama

    @_cache_result
    def _rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @_cache_result
    def _macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}

    @_cache_result
    def _atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    @_cache_result
    def _custom_indicator1(self, data: pd.DataFrame) -> pd.Series:
        # Exemple d'indicateur personnalisé
        return data['close'].rolling(20).mean() / data['close'].rolling(100).mean()

    @_cache_result
    def _custom_indicator2(self, data: pd.DataFrame) -> pd.Series:
        # Autre exemple d'indicateur personnalisé
        return data['volume'].rolling(20).mean() / data['volume'].rolling(50).mean()

    def _order_imbalance(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        if 'buy_orders' in data.columns and 'sell_orders' in data.columns:
            imbalance = (data['buy_orders'] - data['sell_orders']) / (data['buy_orders'] + data['sell_orders'])
            return imbalance.rolling(period).mean()
        return pd.Series(np.nan, index=data.index)
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from talib import abstract
from datetime import datetime

class AdvancedIndicators:
    def __init__(self):
        self.indicators = {
            'trend': self._trend_indicators(),
            'momentum': self._momentum_indicators(),
            'volatility': self._volatility_indicators(),
            'volume': self._volume_indicators(),
            'orderflow': self._orderflow_indicators(),
            'custom': self._custom_indicators(),
            'advanced': self._advanced_indicators()
        }
        self.cache = {}
        self.version = "2.0.0"
        self.last_update = datetime.now()

    def _trend_indicators(self) -> Dict:
        return {
            'ichimoku': self._ichimoku,
            'supertrend': self._supertrend,
            'vwma': self._vwma,
            'trix': self._trix,
            'dema': self._dema,
            'tema': self._tema,
            'kama': self._kama,
            'hull_ma': self._hull_ma,
            'zlema': self._zero_lag_ema,
            'mcginley': self._mcginley,
            'vidya': self._vidya,
            'tma': self._triangular_ma
        }

    def _momentum_indicators(self) -> Dict:
        return {
            'rsi': self._rsi,
            'stoch_rsi': self._stoch_rsi,
            'macd': self._macd,
            'cci': self._cci,
            'williams_r': self._williams_r,
            'ultimate_oscillator': self._ultimate_oscillator,
            'ao': self._awesome_oscillator,
            'ppo': self._ppo,
            'kst': self._kst,
            'dm': self._dm_index,
            'elder_ray': self._elder_ray,
            'psar': self._parabolic_sar
        }

    def _volatility_indicators(self) -> Dict:
        return {
            'atr': self._atr,
            'bb_width': self._bollinger_bandwidth,
            'keltner': self._keltner_channels,
            'parkinson': self._parkinson_volatility,
            'yang_zhang': self._yang_zhang_volatility,
            'chaikin_volatility': self._chaikin_volatility,
            'normalized_atr': self._norm_atr,
            'ulcer_index': self._ulcer_index,
            'chaos_bands': self._chaos_bands,
            'vortex': self._vortex
        }

    def _volume_indicators(self) -> Dict:
        return {
            'obv': self._obv,
            'vwap': self._vwap,
            'accumulation': self._accumulation_distribution,
            'cmf': self._chaikin_money_flow,
            'mfi': self._money_flow_index,
            'eom': self._ease_of_movement,
            'volume_profile': self._volume_profile
        }

    def _orderflow_indicators(self) -> Dict:
        return {
            'bid_ask_ratio': self._bid_ask_ratio,
            'liquidity_wave': self._liquidity_wave,
            'smart_money_index': self._smart_money_index,
            'delta_volume': self._delta_volume,
            'imbalance': self._order_imbalance,
            'order_flow_score': self._order_flow_score,
            'cvd': self._cumulative_volume_delta
        }

    def _advanced_indicators(self) -> Dict:
        return {
            'hurst': self._hurst_exponent,
            'fractal': self._fractal_dimension,
            'entropy': self._price_entropy,
            'correlation_dimension': self._correlation_dimension,
            'lyapunov': self._lyapunov_exponent
        }

    def _custom_indicators(self) -> Dict:
        return {
            'rvi': self._relative_vigor_index,
            'psychological_line': self._psychological_line
        }

    def calculate_all(self, data: pd.DataFrame) -> Dict:
        results = {}
        for category, indicators in self.indicators.items():
            results[category] = {
                name: func(data) 
                for name, func in indicators.items()
            }
        return results

    def _cache_result(func):
        def wrapper(self, data, *args, **kwargs):
            key = f"{func.__name__}_{hash(str(data.index[-1]))}"
            if key not in self.cache:
                self.cache[key] = func(self, data, *args, **kwargs)
            return self.cache[key]
        return wrapper

    @_cache_result
    def _ichimoku(self, data: pd.DataFrame) -> Dict:
        tenkan = self._calc_midpoint(data, 9)
        kijun = self._calc_midpoint(data, 26)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = self._calc_midpoint(data, 52)
        chikou = data['close'].shift(-26)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }

    def _calc_midpoint(self, data: pd.DataFrame, period: int) -> pd.Series:
        high = data['high'].rolling(period).max()
        low = data['low'].rolling(period).min()
        return (high + low) / 2

    @_cache_result
    def _supertrend(self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict:
        """Calculate SuperTrend indicator"""
        atr = self._atr(data, period)
        hl2 = (data['high'] + data['low']) / 2
        
        upper_basic = hl2 + (multiplier * atr)
        lower_basic = hl2 - (multiplier * atr)
        
        upper_band = pd.Series(index=data.index, dtype='float64')
        lower_band = pd.Series(index=data.index, dtype='float64')
        
        for i in range(len(data)):
            if i == 0:
                upper_band[i] = upper_basic[i]
                lower_band[i] = lower_basic[i]
            else:
                upper_band[i] = upper_basic[i] if data['close'][i-1] > upper_band[i-1] else min(upper_basic[i], upper_band[i-1])
                lower_band[i] = lower_basic[i] if data['close'][i-1] < lower_band[i-1] else max(lower_basic[i], lower_band[i-1])
        
        return {
            'upper': upper_band,
            'lower': lower_band,
            'trend': pd.Series(np.where(data['close'] > upper_band, 1, -1), index=data.index)
        }

    @_cache_result
    def _vwma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Moving Average"""
        return (data['close'] * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()

    @_cache_result
    def _trix(self, data: pd.DataFrame, period: int = 18) -> pd.Series:
        """Calculate TRIX indicator"""
        ema1 = data['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100

    @_cache_result
    def _dema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Double Exponential Moving Average"""
        ema = data['close'].ewm(span=period, adjust=False).mean()
        return 2 * ema - ema.ewm(span=period, adjust=False).mean()

    @_cache_result
    def _tema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Triple Exponential Moving Average"""
        ema1 = data['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    @_cache_result
    def _kama(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Kaufman Adaptive Moving Average"""
        change = abs(data['close'] - data['close'].shift(period))
        volatility = abs(data['close'] - data['close'].shift(1)).rolling(period).sum()
        er = change / volatility
        fastest = 2 / (2 + 1)
        slowest = 2 / (30 + 1)
        sc = (er * (fastest - slowest) + slowest) ** 2
        
        kama = pd.Series(index=data.index, dtype='float64')
        kama.iloc[period-1] = data['close'].iloc[period-1]
        for i in range(period, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data['close'].iloc[i] - kama.iloc[i-1])
        return kama

    @_cache_result
    def _hull_ma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Hull Moving Average"""
        wma1 = data['close'].rolling(period // 2).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
        wma2 = data['close'].rolling(period).apply(lambda x: np.average(x, weights=range(1, len(x) + 1)))
        diff = 2 * wma1 - wma2
        return diff.rolling(int(np.sqrt(period))).mean()

    @_cache_result
    def _zero_lag_ema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Zero-Lag Exponential Moving Average"""
        lag = (period - 1) // 2
        return data['close'] + (data['close'] - data['close'].shift(lag))

    @_cache_result
    def _mcginley(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate McGinley Dynamic"""
        mg = pd.Series(data['close'], name='mg')
        mg[0] = data['close'][0]
        for i in range(1, len(data)):
            mg[i] = mg[i-1] + (data['close'][i] - mg[i-1]) / (period * pow(data['close'][i] / mg[i-1], 4))
        return mg
  @_cache_result
    def _rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @_cache_result
    def _stoch_rsi(self, data: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict:
        """Calculate Stochastic RSI"""
        rsi = self._rsi(data, period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        k = stoch_rsi.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return {'k': k, 'd': d}

    @_cache_result
    def _macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence/Divergence)"""
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': macd - signal_line
        }

    @_cache_result
    def _cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (data['high'] + data['low'] + data['close']) / 3
        tp_sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - tp_sma) / (0.015 * mad)

    @_cache_result
    def _williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data['high'].rolling(period).max()
        lowest_low = data['low'].rolling(period).min()
        return ((highest_high - data['close']) / (highest_high - lowest_low)) * -100

    @_cache_result
    def _ultimate_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator"""
        bp = data['close'] - pd.Series(np.minimum(data['low'], data['close'].shift(1)))
        tr = pd.Series(np.maximum(data['high'], data['close'].shift(1))) - \
             pd.Series(np.minimum(data['low'], data['close'].shift(1)))
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

    @_cache_result
    def _awesome_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Awesome Oscillator"""
        median_price = (data['high'] + data['low']) / 2
        return median_price.rolling(5).mean() - median_price.rolling(34).mean()
@_cache_result
    def _ppo(self, data: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
        """Price Percentage Oscillator"""
        fast_ema = data['close'].ewm(span=fast).mean()
        slow_ema = data['close'].ewm(span=slow).mean()
        return ((fast_ema - slow_ema) / slow_ema) * 100

    @_cache_result
    def _kst(self, data: pd.DataFrame) -> pd.Series:
        """Know Sure Thing"""
        rcma1 = self._rcma(data, 10, 10)
        rcma2 = self._rcma(data, 15, 10)
        rcma3 = self._rcma(data, 20, 10)
        rcma4 = self._rcma(data, 30, 15)
        return rcma1 * 1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4

    def _rcma(self, data: pd.DataFrame, roc_period: int, ma_period: int) -> pd.Series:
        """Rate of Change Moving Average - Helper for KST"""
        roc = ((data['close'] - data['close'].shift(roc_period)) / 
               data['close'].shift(roc_period)) * 100
        return roc.rolling(ma_period).mean()

    @_cache_result
    def _dm_index(self, data: pd.DataFrame, period: int = 14) -> Dict:
        """Directional Movement Index"""
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        
        tr = self._atr(data, 1)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return {'plus_di': plus_di, 'minus_di': minus_di, 'adx': adx}

    @_cache_result
    def _elder_ray(self, data: pd.DataFrame, period: int = 13) -> Dict:
        """Elder Ray Index"""
        ema = data['close'].ewm(span=period).mean()
        bull_power = data['high'] - ema
        bear_power = data['low'] - ema
        return {'bull_power': bull_power, 'bear_power': bear_power}

    @_cache_result
    def _obv(self, data: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        obv = pd.Series(0, index=data.index)
        obv[0] = data['volume'][0]
        for i in range(1, len(data)):
            if data['close'][i] > data['close'][i-1]:
                obv[i] = obv[i-1] + data['volume'][i]
            elif data['close'][i] < data['close'][i-1]:
                obv[i] = obv[i-1] - data['volume'][i]
            else:
                obv[i] = obv[i-1]
        return obv

    @_cache_result
    def _vwap(self, data: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        return ((data['high'] + data['low'] + data['close']) / 3 * 
                data['volume']).cumsum() / data['volume'].cumsum()

    @_cache_result
    def _accumulation_distribution(self, data: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        return (clv * data['volume']).cumsum()

    @_cache_result
    def _bid_ask_ratio(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Bid/Ask Ratio"""
        if 'bid_volume' in data.columns and 'ask_volume' in data.columns:
            ratio = data['bid_volume'] / data['ask_volume']
            return ratio.rolling(period).mean()
        return pd.Series(np.nan, index=data.index)

    @_cache_result
    def _liquidity_wave(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Liquidity Wave"""
        if all(col in data.columns for col in ['bid_size', 'ask_size']):
            wave = (data['bid_size'] - data['ask_size']) / (data['bid_size'] + data['ask_size'])
            return wave.rolling(period).mean()
        return pd.Series(np.nan, index=data.index)

    @_cache_result
    def _smart_money_index(self, data: pd.DataFrame) -> pd.Series:
        """Smart Money Index"""
        first_hour = data['close'].shift(1) - data['open']
        last_hour = data['close'] - data['close'].shift(1)
        return (first_hour - last_hour).cumsum()

    @_cache_result
    def _delta_volume(self, data: pd.DataFrame) -> pd.Series:
        """Delta Volume"""
        if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
            return data['buy_volume'] - data['sell_volume']
        return pd.Series(np.nan, index=data.index)

    @_cache_result
    def _order_flow_score(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Order Flow Score"""
        if all(col in data.columns for col in ['bid_volume', 'ask_volume', 'bid_size', 'ask_size']):
            score = ((data['bid_volume'] * data['bid_size']) - 
                    (data['ask_volume'] * data['ask_size'])) / \
                   (data['bid_volume'] * data['bid_size'] + 
                    data['ask_volume'] * data['ask_size'])
            return score.rolling(period).mean()
        return pd.Series(np.nan, index=data.index)

    @_cache_result
    def _hurst_exponent(self, data: pd.DataFrame, min_k: int = 2, max_k: int = 20) -> pd.Series:
        """Hurst Exponent"""
        prices = data['close'].values
        hurst_series = pd.Series(index=data.index)
        
        for i in range(len(prices) - max_k):
            price_section = prices[i:i+max_k]
            lags = range(min_k, max_k)
            tau = [np.sqrt(np.std(np.subtract(price_section[lag:], price_section[:-lag]))) 
                  for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst_series.iloc[i] = poly[0] * 2.0
            
        return hurst_series

    @_cache_result
    def _fractal_dimension(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Fractal Dimension"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        n1 = tr.rolling(period).sum()
        n2 = high_low.rolling(period).sum()
        
        return np.log(n1) / np.log(n2)

    @_cache_result
    def _price_entropy(self, data: pd.DataFrame, period: int = 20, bins: int = 10) -> pd.Series:
        """Price Entropy"""
        def entropy(x):
            hist, _ = np.histogram(x, bins=bins)
            prob = hist / len(x)
            return -np.sum(prob * np.log2(prob + 1e-9))
            
        return data['close'].rolling(period).apply(entropy)

    @_cache_result
    def _relative_vigor_index(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Relative Vigor Index"""
        close = data['close']
        open_price = data['open']
        high = data['high']
        low = data['low']
        
        a = (close - open_price).rolling(period).mean()
        b = (high - low).rolling(period).mean()
        
        return a / b

    @_cache_result
    def _psychological_line(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Psychological Line"""
        diff = data['close'].diff()
        return diff.apply(lambda x: 1 if x > 0 else 0).rolling(period).mean() * 100
