# trading_analysis/models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import BigInteger, Boolean, Column, DateTime, String, Integer, Float, PrimaryKeyConstraint, Text

Base = declarative_base()

class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True)
    interval = Column(String, index=True)
    timestamp = Column(BigInteger, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    interval = Column(String, index=True)
    date = Column(DateTime, index=True)
    params_json = Column(Text)
    loss = Column(Float)
    pnl = Column(Float)
    total_trades = Column(Integer)
    winrate = Column(Float)
    risk_pct = Column(Float)
    retrained = Column(Boolean)
