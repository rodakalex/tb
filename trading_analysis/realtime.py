# trading_analysis/realtime.py
import hashlib
import json
from pathlib import Path
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime, timedelta, timezone

from strategy.utils_hashing import hash_dataframe, hash_params
from trading_analysis.signals import generate_signals_cached
from trading_analysis.bybit_api import get_bybit_kline
from trading_analysis.indicators import calculate_indicators_cached
from trading_analysis.db import (
    save_ohlcv_to_db,
    load_ohlcv_from_db,
    get_latest_timestamp,
)
from trading_analysis.utils import sanitize_params

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
    df = get_bybit_kline(symbol, interval, limit)
    save_ohlcv_to_db(df=df, symbol=symbol)

    df = load_ohlcv_from_db(symbol, limit)
    live_candle = get_live_kline(symbol)
    df = merge_live_candle(df, symbol, live_candle, interval_minutes=interval)

    # Хэшируем данные
    df_hash = hash_dataframe(df)

    # Загрузка и применение параметров
    params = load_best_params(symbol)
    if params:
        sanitized_params = sanitize_params(params)
        params_hash = hash_params(sanitized_params)

        df = calculate_indicators_cached(df_hash, params_hash, df, sanitized_params)
        df = generate_signals_cached(df, json.dumps(sanitized_params, sort_keys=True))
    else:
        print(f"[WARN] No optimized params for {symbol}, skipping signal generation")
        # Даже без params можно считать индикаторы, если хочешь
        df = calculate_indicators_cached(df_hash, "noparams", df, {})

    return df

