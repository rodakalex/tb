# trading_analysis/db.py
import json
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from trading_analysis.bybit_api import find_first_kline_timestamp, get_bybit_kline
from trading_analysis.models import Base, Candle, ModelRun

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

def get_latest_timestamp(symbol: str, interval: str = "30") -> int:
    session = SessionLocal()
    result = session.query(func.max(Candle.timestamp)).filter(
        Candle.symbol == symbol,
        Candle.interval == interval
    ).scalar()
    session.close()
    return result or 0


def fetch_and_save_all_ohlcv(symbol: str, interval: str = "30", batch_limit: int = 1000):
    print(f"📦 Загружаем ВСЕ исторические свечи для {symbol} ({interval}m)...")
    start_ts = find_first_kline_timestamp(symbol, interval)
    now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    ms_per_candle = int(interval) * 60 * 1000
    total_candles = (now_ts - start_ts) // ms_per_candle

    total_saved = 0
    last_ts_seen = None

    with tqdm(total=total_candles, desc=f"{symbol} {interval}m", unit="свеч", ncols=100) as pbar:
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

            start_ts = last_ts + 1
            pbar.update(len(df))

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

def safe_check_ohlcv_integrity(symbol: str, interval: str = '30'):
    """
    Безопасная проверка целостности данных:
    - наличие свечей в БД
    - равномерность интервалов
    - свежесть последней свечи
    """
    print(f"🧪 Проверка целостности данных для {symbol} ({interval})...")

    try:
        df = load_ohlcv_from_db(symbol, limit=100_000, interval=interval)
    except ValueError as e:
        print(f"❌ Ошибка загрузки OHLCV: {e}")
        return False

    if df.empty:
        print("❌ Нет данных в БД.")
        return False

    try:
        expected_delta = interval_to_timedelta(interval)
    except ValueError as e:
        print(f"❌ Ошибка интервала: {e}")
        return False

    # Проверка на пропущенные свечи
    timestamps = df.index.to_list()
    missing = [
        prev + expected_delta * (i + 1)
        for prev, curr in zip(timestamps[:-1], timestamps[1:])
        if (delta := curr - prev) != expected_delta
        for i in range(int(delta / expected_delta) - 1)
    ]

    if missing:
        print(f"⚠️ Обнаружены пропущенные свечи: {len(missing)}")
        for ts in missing[:10]:
            print(f" - {ts}")
        if len(missing) > 10:
            print("...и ещё", len(missing) - 10)
        return False
    else:
        print("✅ Все свечи на месте и интервал целостен.")

    # Проверка свежести последней свечи
    now = datetime.now(timezone.utc)
    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    if now - last_ts > expected_delta:
        print(f"⚠️ Последняя свеча отстаёт: {last_ts} < {now} (на {now - last_ts})")
        return False
    else:
        print("✅ Последняя свеча свежая.")

    return True

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    return obj

def save_model_run(symbol, interval, date, params, loss, pnl, total_trades, winrate, risk_pct, retrained, triggered_restart):
    session = SessionLocal()
    params_clean = convert_np(params)

    run = ModelRun(
        symbol=symbol,
        interval=interval,
        date=date,
        params_json=json.dumps(params_clean),
        loss=loss,
        pnl=pnl,
        total_trades=total_trades,
        winrate=winrate,
        risk_pct=risk_pct,
        retrained=retrained,
        triggered_restart=triggered_restart
    )
    session.add(run)
    session.commit()
    session.close()
    
def get_first_candle_from_db(symbol: str, interval: str = "30") -> pd.Series:
    """
    Возвращает первую (самую раннюю) доступную свечу (OHLCV) из базы данных.
    """
    session = SessionLocal()
    try:
        stmt = (
            select(Candle)
            .where(Candle.symbol == symbol, Candle.interval == interval)
            .order_by(Candle.timestamp.asc())
            .limit(1)
        )
        candle = session.execute(stmt).scalar_one_or_none()

        if candle is None:
            raise ValueError(f"❌ В базе нет свечей для {symbol} ({interval})")

        return pd.Series({
            "timestamp": datetime.fromtimestamp(candle.timestamp / 1000),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume
        })

    finally:
        session.close()

def ensure_data_loaded(symbol: str = "BTCUSDT", interval: str = "30m"):
    print(f"[DEBUG] 🔎 Проверка данных для {symbol}, интервал: {interval}")
    init_db()
    print("[DEBUG] ✅ База инициализирована")
    
    has_data = safe_check_ohlcv_integrity(symbol=symbol, interval=interval)
    print(f"[DEBUG] 📊 Данные найдены? {'Да' if has_data else 'Нет'}")
