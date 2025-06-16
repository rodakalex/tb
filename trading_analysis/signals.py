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

@register_signal("macd", long=True)
def long_macd(df):
    if "MACD" in df and "Signal" in df:
        return ((df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) <= df["Signal"].shift(1)) & (df["MACD"] > 0)).astype(int)
    return pd.Series(0, index=df.index)

@register_signal("macd", long=False)
def short_macd(df):
    if "MACD" in df and "Signal" in df:
        return ((df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) >= df["Signal"].shift(1)) & (df["MACD"] < 0)).astype(int)
    return pd.Series(0, index=df.index)

@register_signal("rsi", long=True)
def long_rsi(df):
    if "RSI" in df:
        return ((df["RSI"] > 45) & (df["RSI"] < 75)).astype(int)
    return pd.Series(0, index=df.index)

@register_signal("rsi", long=False)
def short_rsi(df):
    return (df["RSI"] < 35).astype(int)

@register_signal("roc", long=True)
def long_roc(df):
    if "ROC" in df:
        return (df["ROC"] > 1).astype(int)
    return pd.Series(0, index=df.index)

@register_signal("roc", long=False)
def short_roc(df):
    if "ROC" in df:
        return (df["ROC"] < -1).astype(int)
    return pd.Series(0, index=df.index)

@register_signal("mfi", long=False)
def short_mfi(df):
    return (df["MFI"] > 70).astype(int)

@register_signal("cci", long=False)
def short_cci(df):
    return (df["CCI"] > 100).astype(int)

@register_signal("bb_rebound", long=False)
def short_bb_rebound(df):
    return (df["close"] > df["BBM"]).astype(int)

@register_signal("below_ema9", long=False)
def short_below_ema9(df):
    return (df["close"] < df["EMA_9"]).astype(int)

@register_signal("donchian", long=False)
def short_donchian(df):
    return (df["close"] < df["low"].rolling(20, min_periods=1).min()).astype(int)

@register_signal("tema_cross", long=False)
def short_tema_cross(df):
    return (df["TEMA_9"] < df["TEMA_21"]).astype(int)

@register_signal("stochrsi", long=False)
def short_stochrsi(df):
    if "StochRSI_K" not in df or "StochRSI_D" not in df:
        return None
    return ((df["StochRSI_K"] > 80) & (df["StochRSI_D"] > 80)).astype(int)


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

    df["ema200_down"] = (df.get("EMA_200").diff() < 0 if "EMA_200" in df else False).astype(int)
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


    # if verbose:
    #     print("📋 Сигналы на вход (long):", long_signals)


    
    if p("use_trend_filter", True) and "trend_strength" in df:
        df["long_entry"] &= df["trend_strength"] == 1
    if p("use_ema200_down_filter", True):
        df["short_entry"] &= df["ema200_down"] == 1

    return df
