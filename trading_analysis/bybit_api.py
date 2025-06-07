# trading_analysis/bybit_api.py
from pybit.unified_trading import HTTP
import requests
import pandas as pd
from typing import Optional
import asyncio
from datetime import datetime, timedelta, timezone

from trading_analysis.db import save_ohlcv_to_db

bybit_client = HTTP(testnet=False)
candle_buffer = {}

def get_bybit_trading_symbols(category: str = 'linear') -> list[str]:
    """
    Получает список торговых пар с Bybit по выбранной категории.

    :param category: 'spot', 'linear', 'inverse'
    :return: список тикеров
    """
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": category}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['retCode'] == 0:
            return [item['symbol'] for item in data['result']['list']]
        else:
            print(f"Ошибка API: {data['retMsg']}")
            return []
    except Exception as e:
        print(f"Ошибка при получении тикеров: {e}")
        return []


def fetch_bybit_kline_raw(
    symbol: str,
    interval: str = "30",
    limit: int = 1000,
    start: Optional[int] = None,
    end: Optional[int] = None,
    category: str = "linear"
) -> list[list[str]]:
    """
    Получает торговые свечи через pybit (Bybit HTTP Unified API).
    """
    try:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        if start:
            params["start"] = start
        if end:
            params["end"] = end

        res = bybit_client.get_kline(**params)

        if "result" in res and "list" in res["result"]:
            return res["result"]["list"]
        else:
            raise ValueError(f"Некорректный ответ от pybit: {res}")

    except Exception as e:
        raise RuntimeError(f"Ошибка при получении свечей через pybit: {e}")



def parse_kline_to_df(raw_klines: list[list[str]]) -> pd.DataFrame:
    """
    Преобразует сырые свечи в DataFrame с объёмом.

    Формат:
    [timestamp, open, high, low, close, volume, turnover]

    :param raw_klines: список свечей
    :return: DataFrame
    """
    columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(raw_klines, columns=columns)

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='ms')

    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
        "turnover": float
    })

    df.set_index("timestamp", inplace=True)

    return df.sort_index()

def get_bybit_kline(
    symbol: str,
    interval: str = "30",
    limit: int = 1000,
    start: Optional[int] = None,
    end: Optional[int] = None
) -> pd.DataFrame:
    """
    Получает и возвращает торговые свечи с объёмом в виде DataFrame.

    :param symbol: тикер
    :param interval: интервал свечей
    :param limit: ограничение по количеству
    :param start: начало диапазона (мс)
    :param end: конец диапазона (мс)
    :return: DataFrame с OHLCV
    """
    raw = fetch_bybit_kline_raw(symbol, interval, limit, start, end)
    return parse_kline_to_df(raw)

async def periodically_update_history(symbol: str, interval_minutes: int = 30):
    while True:
        print(f"[TASK] Обновляем историю для {symbol}")
        df = get_bybit_kline(symbol, str(interval_minutes), limit=1000)
        save_ohlcv_to_db(df, symbol)
        await asyncio.sleep(interval_minutes * 60)

def find_first_kline_timestamp(symbol: str, interval: str = "30") -> int:
    print(f"🔍 Начинаем поиск самой ранней свечи для {symbol} с интервалом {interval}m...")

    now = datetime.now(timezone.utc)
    interval_minutes = int(interval)
    delta = timedelta(minutes=interval_minutes * 1000)
    ts = now

    checked_batches = 0

    while True:
        ts -= delta
        start_ms = int(ts.timestamp() * 1000)
        data = fetch_bybit_kline_raw(symbol=symbol, interval=interval, limit=1, start=start_ms)

        if not data:
            break

        # Проверяем: если свеча новее, чем мы запросили — значит дальше смысла нет
        first_ts = int(data[0][0])  # самая новая свеча
        first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)

        if first_dt > ts + timedelta(minutes=interval_minutes):
            break

        checked_batches += 1
        if checked_batches % 20 == 0:
            print(f"⏪ Проверено {checked_batches} батчей. Текущая дата: {ts.strftime('%Y-%m-%d %H:%M')}")

    # Выход: возвращаем timestamp найденной самой старой свечи
    confirmed_ts = int(ts.timestamp() * 1000)
    confirmed_data = fetch_bybit_kline_raw(symbol=symbol, interval=interval, limit=1, start=confirmed_ts)
    
    if confirmed_data:
        final_ts = int(confirmed_data[0][0])
        print("✅ Первая доступная свеча:", datetime.fromtimestamp(final_ts / 1000, tz=timezone.utc))
        return final_ts
    else:
        raise RuntimeError("❌ Не удалось найти первую свечу")

def load_1000_df(symbol="PRIMEUSDT", interval="30", start_ts: Optional[int] = None):
    if start_ts is None:
        start_ts = find_first_kline_timestamp(symbol, interval)
    df = get_bybit_kline(symbol=symbol, interval=interval, limit=1000, start=start_ts)
    return df
