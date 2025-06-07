# trading_analysis/indicators.py
import pandas as pd

def safe_assign_ema(df, length, col_name):
    result = df.ta.ema(length=length)
    if isinstance(result, pd.DataFrame):
        df[col_name] = result.iloc[:, 0]
    else:
        df[col_name] = result
    return df


def calculate_indicators(df):
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float
    })

    macd = df.ta.macd(fast=12, slow=26, signal=9)
    macd.columns = ["MACD", "Signal", "Histogram"]

    rsi = df.ta.rsi(length=14)
    df["RSI"] = rsi
    df = safe_assign_ema(df, 9, "EMA_9")
    df = safe_assign_ema(df, 21, "EMA_21")
    df = safe_assign_ema(df, 100, "EMA_100")
    df = safe_assign_ema(df, 200, "EMA_200")

    bbands = df.ta.bbands(length=20, std=2)
    bbands.columns = ["BBL", "BBM", "BBU", "BBB", "BBP"]

    df["Volume_SMA_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["ATR"] = df.ta.atr(length=14)
    df = pd.concat([df, macd, bbands], axis=1)

    return df
