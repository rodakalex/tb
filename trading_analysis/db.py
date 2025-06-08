# trading_analysis/db.py
import numpy as np
import json
import time
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from trading_analysis.bybit_api import find_first_kline_timestamp, get_bybit_kline
from trading_analysis.models import Candle, Base, ModelRun
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from trading_analysis.models import Base, Candle
import pandas as pd
from sqlalchemy import func
from datetime import timedelta

engine = create_engine("sqlite:///market_data.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def init_db():
    Base.metadata.create_all(engine)

def save_ohlcv_to_db(df: pd.DataFrame, symbol: str, interval: str):
    if df.empty:
        print(f"[DB] ❌ Пустой DataFrame для {symbol} ({interval}) — ничего не сохраняем.")
        return

    df = df.copy().reset_index()
    df["timestamp"] = (df["timestamp"].astype("int64") // 10**6)

    session = SessionLocal()

    # Получаем уже существующие timestamps для этого символа и интервала
    existing_ts = set(
        ts for (ts,) in session.query(Candle.timestamp)
        .filter(Candle.symbol == symbol)
        .filter(Candle.interval == interval)
        .filter(Candle.timestamp.in_(df["timestamp"].tolist()))
        .all()
    )

    new_rows = df[~df["timestamp"].isin(existing_ts)]

    records = [
        Candle(
            symbol=symbol,
            interval=interval,  # 🆕 Сохраняем интервал
            timestamp=int(row["timestamp"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"]
        )
        for _, row in new_rows.iterrows()
    ]

    if records:
        session.bulk_save_objects(records)
        session.commit()
        print(f"[DB] ✅ Сохранено {len(records)} новых свечей для {symbol} ({interval})")
    else:
        print(f"[DB] 🔁 Нет новых свечей для {symbol} ({interval})")

    session.close()

def load_ohlcv_from_db(
    symbol: str,
    interval: str,
    start_timestamp: int = None,
    end_timestamp: int = None,
    limit: int = None,
) -> pd.DataFrame:
    session = SessionLocal()

    stmt = select(Candle).where(
        Candle.symbol == symbol,
        Candle.interval == interval  # ⬅ фильтрация по таймфрейму
    )

    if start_timestamp is not None:
        stmt = stmt.where(Candle.timestamp >= start_timestamp)
    if end_timestamp is not None:
        stmt = stmt.where(Candle.timestamp <= end_timestamp)

    stmt = stmt.order_by(Candle.timestamp.asc())

    if limit is not None:
        stmt = stmt.limit(limit)

    candles = session.execute(stmt).scalars().all()
    session.close()

    if not candles:
        raise ValueError(f"No OHLCV data found for {symbol} ({interval}) with given parameters.")

    data = [{
        "timestamp": c.timestamp,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "volume": c.volume
    } for c in candles]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df

def get_latest_timestamp(symbol: str) -> int:
    session = SessionLocal()
    result = session.query(func.max(Candle.timestamp)).filter(Candle.symbol == symbol).scalar()
    session.close()
    return result or 0

def fetch_and_save_all_ohlcv(symbol: str, interval: str = "30", batch_limit: int = 1000):
    print(f"📦 Загружаем ВСЕ исторические свечи для {symbol} ({interval}m)...")
    start_ts = find_first_kline_timestamp(symbol, interval)
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    total_saved = 0
    last_ts_seen = None

    while start_ts < now_ts:
        df = get_bybit_kline(symbol, interval, limit=batch_limit, start=start_ts)
        if df.empty:
            print("❌ Получен пустой DataFrame — остановка.")
            break

        last_ts = int(df.index[-1].timestamp() * 1000)
        if last_ts == last_ts_seen:
            print("🛑 Повтор последней свечи — остановка.")
            break
        last_ts_seen = last_ts

        save_ohlcv_to_db(df, symbol, interval=interval)
        total_saved += len(df)

        start_ts = last_ts + 1  # следующий после последнего
        print(f"🔄 Загружено и сохранено ещё {len(df)} свечей. Продолжаем...")

        time.sleep(0.25)

    print(f"\n✅ Завершено. Всего сохранено свечей: {total_saved}")

def interval_to_timedelta(interval: str) -> timedelta:
    """
    Преобразует интервал Bybit в timedelta.
    Примеры: '1', '3', ..., '720', 'D', 'W', 'M'
    """
    if interval.isdigit():
        return timedelta(minutes=int(interval))
    elif interval == 'D':
        return timedelta(days=1)
    elif interval == 'W':
        return timedelta(weeks=1)
    elif interval == 'M':
        return timedelta(days=30)  # усреднённая длина месяца
    else:
        raise ValueError(f"Неизвестный формат интервала Bybit: '{interval}'")

def check_ohlcv_integrity(symbol: str, interval: str = '30'):
    print(f"🧪 Проверка целостности данных для {symbol} ({interval})...")
    df = load_ohlcv_from_db(symbol, limit=100_000, interval=interval)

    if df.empty:
        print("❌ Нет данных в БД.")
        return

    try:
        expected_delta = interval_to_timedelta(interval)
    except ValueError as e:
        print(f"❌ {e}")
        return

    missing_timestamps = []

    timestamps = df.index.to_list()
    for prev, curr in zip(timestamps[:-1], timestamps[1:]):
        delta = curr - prev
        if delta != expected_delta:
            missing_count = int(delta / expected_delta) - 1
            for i in range(missing_count):
                missing_ts = prev + expected_delta * (i + 1)
                missing_timestamps.append(missing_ts)

    if missing_timestamps:
        print(f"⚠️ Обнаружены пропущенные свечи: {len(missing_timestamps)}")
        for ts in missing_timestamps[:10]:
            print(f" - {ts}")
        if len(missing_timestamps) > 10:
            print("...и ещё", len(missing_timestamps) - 10)
    else:
        print("✅ Все свечи присутствуют и интервал целостен.")


def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    return obj

def save_model_run(symbol, date, params, loss, pnl, total_trades, winrate, risk_pct, retrained):
    session = SessionLocal()
    params_clean = convert_np(params)

    run = ModelRun(
        symbol=symbol,
        date=date,
        params_json=json.dumps(params_clean),
        loss=loss,
        pnl=pnl,
        total_trades=total_trades,
        winrate=winrate,
        risk_pct=risk_pct,
        retrained=retrained
    )
    session.add(run)
    session.commit()
    session.close()
    