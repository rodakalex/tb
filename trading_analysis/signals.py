# trading_analysis/signals.py
import numpy as np
import pandas as pd
import pandas_ta as ta
import json
from trading_analysis.cache import memory

LONG_SIGNAL_FUNCS = {}
SHORT_SIGNAL_FUNCS = {}

verbose = False

def register_signal(name, long=True):
    def decorator(func):
        (LONG_SIGNAL_FUNCS if long else SHORT_SIGNAL_FUNCS)[name] = func
        return func
    return decorator

@register_signal("supertrend", long=True)
def long_supertrend(df):
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["SUPERTREND"] = st["SUPERT_10_3.0"]
    return (df["close"] > df["SUPERTREND"]).astype(int)

@register_signal("supertrend", long=False)
def short_supertrend(df):
    return (df["close"] < df["SUPERTREND"]).astype(int)

@register_signal("volatility_breakout", long=True)
def long_volatility_breakout(df):
    range_ = df["high"].shift(1) - df["low"].shift(1)
    breakout = df["open"] > df["open"].shift(1) + range_ * 0.5
    return breakout.astype(int)

@register_signal("volatility_breakout", long=False)
def short_volatility_breakout(df):
    range_ = df["high"].shift(1) - df["low"].shift(1)
    breakdown = df["open"] < df["open"].shift(1) - range_ * 0.5
    return breakdown.astype(int)

@register_signal("trend_up", long=True)
def long_trend_up(df):
    trend = df["close"].ewm(span=50, adjust=False).mean()
    return (trend.diff() > 0).astype(int)

@register_signal("trend_down", long=False)
def short_trend_down(df):
    trend = df["close"].ewm(span=50, adjust=False).mean()
    return (trend.diff() < 0).astype(int)

@register_signal("atr_breakout", long=True)
def long_atr_breakout(df):
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    return (df["close"].diff() > atr).astype(int)

@register_signal("atr_breakout", long=False)
def short_atr_breakout(df):
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    return (df["close"].diff() < -atr).astype(int)

@register_signal("smart_macd", long=True)
def long_smart_macd(df):
    if "SMART_MACD" not in df or "SMART_SIGNAL" not in df:
        df = smart_macd(df)
    return (
        (df["SMART_MACD"] > df["SMART_SIGNAL"]) &
        (df["SMART_MACD"].shift(1) <= df["SMART_SIGNAL"].shift(1)) &
        (df["SMART_MACD"] > 0)
    ).astype(int)

@register_signal("smart_macd", long=False)
def short_smart_macd(df):
    if "SMART_MACD" not in df or "SMART_SIGNAL" not in df:
        df = smart_macd(df)
    return (
        (df["SMART_MACD"] < df["SMART_SIGNAL"]) &
        (df["SMART_MACD"].shift(1) >= df["SMART_SIGNAL"].shift(1)) &
        (df["SMART_MACD"] < 0)
    ).astype(int)

@register_signal("smart_rsi", long=True)
def long_smart_rsi(df):
    if "SMART_RSI" not in df:
        df["SMART_RSI"] = smart_rsi(df)
    return ((df["SMART_RSI"] > 45) & (df["SMART_RSI"] < 75)).astype(int)

@register_signal("smart_rsi", long=False)
def short_smart_rsi(df):
    if "SMART_RSI" not in df:
        df["SMART_RSI"] = smart_rsi(df)
    return ((df["SMART_RSI"] > 65) & (df["SMART_RSI"] < 90)).astype(int)

@register_signal("below_ema9", long=False)
def short_below_ema9(df):
    return (df["close"] < df["EMA_9"]).astype(int)

@register_signal("smart_trend", long=True)
def long_smart_trend(df):
    if "smart_trend" not in df:
        df = smart_trend_indicator(df)
    return (df["smart_trend"] == 1).astype(int)

@register_signal("smart_trend", long=False)
def short_smart_trend(df):
    if "smart_trend" not in df:
        df = smart_trend_indicator(df)
    return (df["smart_trend"] == -1).astype(int)

def smart_rsi(df, period=14, volume_weighted=True, ema_smooth=True, trend_period=50):
    close = df["close"]
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    if volume_weighted:
        volume = df["volume"]
        gain *= volume
        loss *= volume

    if ema_smooth:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
    else:
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # Трендовая коррекция
    trend = df["close"].ewm(span=trend_period, adjust=False).mean()
    trend_slope = trend.diff()

    trend_adj = trend_slope.apply(lambda x: 2 if x > 0 else -2 if x < 0 else 0)
    smart = rsi + trend_adj

    return smart.clip(0, 100)

def smart_macd(df, fast=8, slow=21, signal=5):
    fast_ma = ta.dema(df["close"], length=fast)
    slow_ma = ta.dema(df["close"], length=slow)
    macd = fast_ma - slow_ma

    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    # Нормализация
    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    norm_histogram = histogram / (atr + 1e-9)

    df["SMART_MACD"] = macd
    df["SMART_SIGNAL"] = signal_line
    df["SMART_HIST"] = norm_histogram
    
    return df

def smart_trend_indicator(df, ema_period=200, atr_period=14, volume_weight=True):
    """
    Более интеллектуальный трендовый индикатор.

    - Если тренд устойчиво восходящий — +1.
    - Если нисходящий — -1.
    - В противном случае — 0 (боковик или слабое движение).
    """

    # 1. Скользящая EMA
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    df["EMA_TREND"] = ema

    # 2. Наклон (slope)
    slope = ema.diff()

    # 3. ATR (волатильность)
    atr = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
    norm_slope = slope / (atr + 1e-9)

    # 4. Взвешивание на объём (опционально)
    if volume_weight:
        vw_norm_slope = norm_slope * (df["volume"] / df["volume"].rolling(20).mean().clip(lower=1))
    else:
        vw_norm_slope = norm_slope

    # 5. Сглаживаем для устойчивости
    smooth = vw_norm_slope.rolling(10).mean()

    # 6. Оценка направления тренда
    trend_signal = (
        (smooth > 0.3).astype(int) -
        (smooth < -0.3).astype(int)
    )

    df["smart_trend"] = trend_signal
    return df

@memory.cache
def generate_signals_cached(df, params_serialized):
    if isinstance(params_serialized, dict):
        params = params_serialized  # поддержка старого кода
    else:
        params = json.loads(params_serialized)
    return generate_signals(df, params)

@register_signal("volume_above_avg", long=True)
def long_volume_above_avg(df):
    return (df["volume"] > df.get("Volume_SMA_20", df["volume"])).astype(int)

def generate_signals(df, params=None):
    if verbose:
        print("use_atr_filter:", params.get("use_atr_filter", True))
    
    import numpy as np
    df = df.copy()
    if params is None:
        params = {}

    def p(key, default=1):
        return params.get(key, default)

    # Бинарные фильтры
    df["volume_above_avg"] = (df["volume"] > df.get("Volume_SMA_20", df["volume"])).astype(int)
    df["atr_filter"] = (
        (df["ATR"] > df["ATR"].rolling(50).mean()) if "ATR" in df
        else pd.Series(False, index=df.index)
    ).astype(int)

    df["volume_peak"] = (df["volume"] > df["volume"].rolling(10).max().shift(1)).astype(int)
    df["bbp_filter"] = (df.get("BBP", 0) > 0.5).astype(int)

    # Сигналы
    long_signals = params.get("enabled_long_signals", list(LONG_SIGNAL_FUNCS))
    short_signals = params.get("enabled_short_signals", list(SHORT_SIGNAL_FUNCS))

    if verbose:
        for name in long_signals:
            if name in LONG_SIGNAL_FUNCS:
                signal_series = LONG_SIGNAL_FUNCS[name](df)
                print(f"📈 Signal {name} sum:", signal_series.sum())
                df[f"long_{name}"] = signal_series


    for name in long_signals:
        if name in LONG_SIGNAL_FUNCS:
            df[f"long_{name}"] = LONG_SIGNAL_FUNCS[name](df)

    for name in short_signals:
        if name in SHORT_SIGNAL_FUNCS:
            signal_series = SHORT_SIGNAL_FUNCS[name](df)
            if signal_series is not None:
                df[f"short_{name}"] = signal_series


    # Score
    df["long_score"] = sum(
        df[f"long_{s}"] * p(f"w_{s}", 1)
        for s in long_signals if f"long_{s}" in df
    )
    df["short_score"] = sum(
        df[f"short_{s}"] * p(f"w_{s}", 1)
        for s in short_signals if f"short_{s}" in df
    )

    # Entry
    df["long_entry"] = df["long_score"] >= p("long_score_threshold", 5)
    df["short_entry"] = df["short_score"] >= p("short_score_threshold", 5)

    if p("use_smart_trend_filter", True) and "smart_trend" in df:
        df["long_entry"] &= df["smart_trend"] == 1
        df["short_entry"] &= df["smart_trend"] == -1


    return df
