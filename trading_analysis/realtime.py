# trading_analysis/realtime.py
import json
from pathlib import Path
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime, timedelta, timezone

from trading_analysis.signals import generate_signals
from trading_analysis.bybit_api import get_bybit_kline, periodically_update_history
from trading_analysis.indicators import calculate_indicators
from trading_analysis.db import (
    save_ohlcv_to_db,
    load_ohlcv_from_db,
    get_latest_timestamp,
)

kline_buffer = defaultdict(lambda: deque(maxlen=1))

def update_live_kline(symbol: str, data: dict):
    ts = pd.to_datetime(int(data["start"]), unit='ms', utc=True)
    kline = {
        "timestamp": ts,
        "open": float(data["open"]),
        "high": float(data["high"]),
        "low": float(data["low"]),
        "close": float(data["close"]),
        "volume": float(data["volume"]),
        "turnover": float(data["turnover"]),
    }
    kline_buffer[symbol].clear()
    kline_buffer[symbol].append(kline)

def get_live_kline(symbol: str) -> dict:
    if symbol in kline_buffer and kline_buffer[symbol]:
        return kline_buffer[symbol][-1]
    return None

def merge_live_candle(df: pd.DataFrame, symbol: str, live_candle: dict, interval_minutes=30) -> pd.DataFrame:
    if not live_candle:
        return df

    ts = live_candle["timestamp"]
    now = datetime.now(timezone.utc)

    df.index = df.index.tz_localize("UTC")
    if ts <= df.index[-1] or now < ts + timedelta(minutes=interval_minutes):
        return df

    print('hello')
    row = pd.DataFrame([{
        "open": live_candle["open"],
        "high": live_candle["high"],
        "low": live_candle["low"],
        "close": live_candle["close"],
        "volume": live_candle["volume"],
        "turnover": live_candle["turnover"]
    }], index=[ts])

    return pd.concat([df, row]).sort_index()

def load_best_params(symbol: str):
    path = Path("best_params") / f"{symbol}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def run_analysis_for_symbol(symbol, interval="30", limit=1000):
    latest_ts = get_latest_timestamp(symbol)

    df = get_bybit_kline(symbol, interval, limit)
    save_ohlcv_to_db(df=df, symbol=symbol)

    df = load_ohlcv_from_db(symbol, limit)
    live_candle = get_live_kline(symbol)
    df = merge_live_candle(df, symbol, live_candle, interval_minutes=interval)

    df = calculate_indicators(df)

    params = load_best_params(symbol)
    if params:
        df = generate_signals(df, params=params)
    else:
        print(f"[WARN] No optimized params for {symbol}, skipping signal generation")

    return df
