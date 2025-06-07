# trading_analysis/models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, Float, PrimaryKeyConstraint

Base = declarative_base()

class Candle(Base):
    __tablename__ = 'candles'
    symbol = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)  # ms
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('symbol', 'timestamp'),
    )
