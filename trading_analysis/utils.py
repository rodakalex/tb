def strip_indicators(df):
    cols_to_drop = [col for col in df.columns if col not in ["open", "high", "low", "close", "volume", "turnover"]]
    return df.drop(columns=cols_to_drop, errors="ignore")

def should_update_data(latest_timestamp_ms: int, interval_minutes: int = 30, tolerance_minutes: int = 1) -> bool:
    last_candle_time = datetime.fromtimestamp(latest_timestamp_ms / 1000, tz=timezone.utc)
    next_expected_time = last_candle_time + timedelta(minutes=interval_minutes)
    now = datetime.now(timezone.utc)
    
    return now >= (next_expected_time - timedelta(minutes=tolerance_minutes))