# trading_analysis/indicators.py
from trading_analysis.cache import memory
import pandas as pd
import pandas_ta as ta


def safe_assign_ema(df, length, col_name):
    result = df.ta.ema(length=length)
    if isinstance(result, pd.DataFrame):
        df[col_name] = result.iloc[:, 0]
    else:
        df[col_name] = result
    return df


def calculate_indicators(df):
    import warnings

    df = df.copy()
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })

    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["Volume_SMA_20"] = df["volume"].rolling(window=20, min_periods=1).mean()

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df[["MACD", "Signal", "Histogram"]] = macd

    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df[["BBL", "BBM", "BBU", "BBB", "BBP"]] = bb

    df["EMA_9"] = ta.ema(df["close"], length=9)
    df["EMA_21"] = ta.ema(df["close"], length=21)
    df["EMA_100"] = ta.ema(df["close"], length=100)
    df["EMA_200"] = ta.ema(df["close"], length=200)

    df["CCI"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    stochrsi = ta.stochrsi(df["close"], length=14)
    if stochrsi is not None and "STOCHRSIk_14_14_3_3" in stochrsi:
        df["StochRSI_K"] = stochrsi["STOCHRSIk_14_14_3_3"]
        df["StochRSI_D"] = stochrsi["STOCHRSId_14_14_3_3"]

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None and "ADX_14" in adx:
        df["ADX"] = adx["ADX_14"]

    df["ROC"] = ta.roc(df["close"], length=5)

    df["TEMA_9"] = ta.tema(df["close"], length=9)
    df["TEMA_21"] = ta.tema(df["close"], length=21)

    return df

@memory.cache(ignore=['df', 'params'])
def calculate_indicators_cached(df_hash: str, df: pd.DataFrame, params: dict):
    return calculate_indicators(df)
