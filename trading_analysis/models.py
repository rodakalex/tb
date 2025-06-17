# trading_analysis/models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, String, Integer, Float, PrimaryKeyConstraint, Text

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
    session_uuid = Column(String(36), index=True, nullable=False)
    symbol = Column(String, index=True)
    interval = Column(String, index=True)
    date = Column(DateTime, index=True)
    best_params = Column(Text)
    balance = Column(Float)
    loss = Column(Float)
    pnl = Column(Float)
    total_trades = Column(Integer)
    winrate = Column(Float)
    risk_pct = Column(Float)
    retrained = Column(Boolean)
    triggered_restart = Column(Boolean, default=False)

    
class TradeLog(Base):
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True)
    session_uuid = Column(
        String(36),
        ForeignKey("model_runs.session_uuid", ondelete="CASCADE"),
        index=True,
        nullable=False
    )
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, nullable=False)

    action = Column(String, nullable=False)   # "entry", "exit", "tp_hit", etc.
    side = Column(String)                     # "long" or "short"

    price = Column(Float)
    qty = Column(Float)
