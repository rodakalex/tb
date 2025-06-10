# trading_analysis/utils.py
import datetime


def strip_indicators(df):
    cols_to_drop = [col for col in df.columns if col not in ["open", "high", "low", "close", "volume", "turnover"]]
    return df.drop(columns=cols_to_drop, errors="ignore")

def should_update_data(latest_timestamp_ms: int, interval_minutes: int = 30, tolerance_minutes: int = 1) -> bool:
    last_candle_time = datetime.fromtimestamp(latest_timestamp_ms / 1000, tz=datetime.timezone.utc)
    next_expected_time = last_candle_time + datetime.timedelta(minutes=interval_minutes)
    now = datetime.now(datetime.timezone.utc)
    
    return now >= (next_expected_time - datetime.timedelta(minutes=tolerance_minutes))

def split_train_val_test(df, val_ratio=0.1, test_ratio=0.1):
    """
    Делит DataFrame на train, val, test по временной оси.
    """
    n = len(df)
    val_size = int(n * val_ratio)
    test_size = int(n * test_ratio)
    train_size = n - val_size - test_size

    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size + val_size]
    df_test = df.iloc[train_size + val_size:]

    return df_train, df_val, df_test
