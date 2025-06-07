# trading_analysis/db.py
from sqlalchemy.orm import Session
from trading_analysis.models import Candle, Base
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from trading_analysis.models import Base, Candle
import pandas as pd
from sqlalchemy import func

engine = create_engine("sqlite:///market_data.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def init_db():
    Base.metadata.create_all(engine)

def save_ohlcv_to_db(df: pd.DataFrame, symbol: str):
    if df.empty:
        return

    df = df.copy()
    df = df.reset_index()
    df["timestamp"] = (df["timestamp"].astype("int64") // 10**6)

    session: Session = SessionLocal()

    existing_ts = set(
        ts for (ts,) in session.query(Candle.timestamp)
        .filter(Candle.symbol == symbol)
        .filter(Candle.timestamp.in_(df["timestamp"].tolist()))
        .all()
    )

    new_rows = df[~df["timestamp"].isin(existing_ts)]

    records = [
        Candle(
            symbol=symbol,
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
        print(f"[DB] Сохранено {len(records)} новых свечей для {symbol}")
    else:
        print(f"[DB] Нет новых свечей для {symbol}")

    session.close()

    
def load_ohlcv_from_db(symbol: str, limit: int = 1000) -> pd.DataFrame:
    session = SessionLocal()
    stmt = (
        select(Candle)
        .where(Candle.symbol == symbol)
        .order_by(Candle.timestamp.desc())
        .limit(limit)
    )
    candles = session.execute(stmt).scalars().all()
    session.close()

    if not candles:
        raise ValueError("No data found")

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
    return df.sort_index()

def get_latest_timestamp(symbol: str) -> int:
    session = SessionLocal()
    result = session.query(func.max(Candle.timestamp)).filter(Candle.symbol == symbol).scalar()
    session.close()
    return result or 0