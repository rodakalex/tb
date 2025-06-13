# trading_analysis/signals.py
import numpy as np
import pandas as pd
import pandas_ta as ta
import json
from trading_analysis.cache import memory

@memory.cache
def generate_signals_cached(df, params_serialized):
    if isinstance(params_serialized, dict):
        params = params_serialized  # поддержка старого кода
    else:
        params = json.loads(params_serialized)
    return generate_signals(df, params)

def safe_assign(df, col_name, series, dtype="float64"):
    df[col_name] = series.astype(dtype)

def generate_signals(df, params=None):
    df = df.copy()
    if params is None:
        params = {}

    def p(key, default=1):
        return params.get(key, default)

    # --- Индикаторы и фильтры ---
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volume_above_avg"] = (df["volume"] > df["Volume_SMA_20"]).astype(int)
    df["atr_filter"] = (df["ATR"] > df["ATR"].rolling(50).mean()).astype(int)
    df["ema200_down"] = (df["EMA_200"].diff() < 0).astype(int)
    df["volume_peak"] = (df["volume"] > df["volume"].rolling(10).max().shift(1)).astype(int)
    df["bbp_filter"] = (df["BBP"] > 0.5).astype(int)

    df["long_macd"] = ((df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) <= df["Signal"].shift(1)) & (df["MACD"] > 0)).astype(int)
    df["short_macd"] = ((df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) >= df["Signal"].shift(1)) & (df["MACD"] < 0) & (df["Signal"] < 0)).astype(int)

    df["long_rsi"] = ((df["RSI"] > 45) & (df["RSI"] < 75)).astype(int)
    df["short_rsi"] = (df["RSI"] < 50).astype(int)

    df["long_ema_cross"] = (df["EMA_9"] > df["EMA_21"]).astype(int)
    df["short_ema_cross"] = (df["EMA_9"] < df["EMA_21"]).astype(int)

    df["long_trend"] = (df["close"] > df["EMA_200"]).astype(int)
    df["short_trend"] = (df["close"] < df["EMA_200"]).astype(int)

    df["short_bb_rebound"] = (df["close"] > df["BBM"]).astype(int)
    df["short_below_ema9"] = (df["close"] < df["EMA_9"]).astype(int)

    df["CCI"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    stochrsi = ta.stochrsi(df["close"], length=14)
    if stochrsi is not None and "STOCHRSIk_14_14_3_3" in stochrsi:
        df["StochRSI_K"] = stochrsi["STOCHRSIk_14_14_3_3"]
        df["StochRSI_D"] = stochrsi["STOCHRSId_14_14_3_3"]
    else:
        df["StochRSI_K"] = np.nan
        df["StochRSI_D"] = np.nan

    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["volume_zscore"] = ((df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()).fillna(0)

    df["long_stochrsi"] = ((df["StochRSI_K"] < 20) & (df["StochRSI_K"] > df["StochRSI_D"])).astype(int)
    df["short_stochrsi"] = ((df["StochRSI_K"] > 80) & (df["StochRSI_K"] < df["StochRSI_D"])).astype(int)
    df["long_cci"] = (df["CCI"] < -100).astype(int)
    df["short_cci"] = (df["CCI"] > 100).astype(int)
    df["long_mfi"] = (df["MFI"] < 30).astype(int)
    df["short_mfi"] = (df["MFI"] > 70).astype(int)
    df["trend_strength"] = (df["ADX"] > 20).astype(int)
    use_custom_roc = p("use_custom_roc", False)
    
    if not use_custom_roc:
        df["ROC"] = ta.roc(df["close"], length=5)
    df["long_roc"] = (df["ROC"] > 1).astype(int)
    df["short_roc"] = (df["ROC"] < -1).astype(int)

    df["volume_spike"] = (df["volume_zscore"] > 2).astype(int)

    donchian_high = df["high"].rolling(window=20).max()
    donchian_low = df["low"].rolling(window=20).min()
    df["long_donchian"] = (df["close"] > donchian_high.shift(1)).astype(int)
    df["short_donchian"] = (df["close"] < donchian_low.shift(1)).astype(int)

    tema_9 = ta.tema(df["close"], length=9)
    tema_21 = ta.tema(df["close"], length=21)

    df["TEMA_9"] = tema_9 if tema_9 is not None else pd.Series([np.nan] * len(df), index=df.index)
    df["TEMA_21"] = tema_21 if tema_21 is not None else pd.Series([np.nan] * len(df), index=df.index)

    df["long_tema_cross"] = (df["TEMA_9"] > df["TEMA_21"]).astype(int)
    df["short_tema_cross"] = (df["TEMA_9"] < df["TEMA_21"]).astype(int)

    # --- Score через enabled_signals ---
    default_long = [
        "long_ema_cross", "long_trend", "long_rsi", "long_macd", "long_stochrsi",
        "long_cci", "long_mfi", "volume_above_avg", "volume_spike",
        "long_roc", "long_donchian", "long_tema_cross"
    ]
    default_short = [
        "short_ema_cross", "short_trend", "short_rsi", "short_macd", "short_stochrsi",
        "short_cci", "short_mfi", "short_bb_rebound", "short_below_ema9",
        "volume_above_avg", "volume_spike", "short_roc", "short_donchian", "short_tema_cross"
    ]

    long_signals = params.get("enabled_long_signals", default_long)
    short_signals = params.get("enabled_short_signals", default_short)

    df["long_score"] = sum(
        df[signal] * p(f"w_{signal.split('long_')[-1]}", 1)
        for signal in long_signals
        if signal in df
    )

    df["short_score"] = sum(
        df[signal] * p(f"w_{signal.split('short_')[-1]}", 1)
        for signal in short_signals
        if signal in df
    )

    # --- Флаги для гибкого управления фильтрами ---
    long_entry = df["long_score"] >= p("long_score_threshold", 5)
    use_atr = p("use_atr_filter", True)
    if use_atr:
        long_entry &= df["atr_filter"] == 1

    use_trend = p("use_trend_filter", True)
    if use_trend:
        long_entry &= df["trend_strength"] == 1

    short_entry = df["short_score"] >= p("short_score_threshold", 5)
    if use_atr:
        short_entry &= df["atr_filter"] == 1
    if p("use_ema200_down_filter", True):
        short_entry &= df["ema200_down"] == 1

    df["long_entry"] = long_entry
    df["short_entry"] = short_entry

    return df
