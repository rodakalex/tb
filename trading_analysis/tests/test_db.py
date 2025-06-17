# trading_analysis/tests/test_model_run.py

import json
import pytest
from trading_analysis.db import save_model_run, SessionLocal
from trading_analysis.models import ModelRun
import pandas as pd
from uuid import uuid4

def test_save_model_run_creates_record():
    session = SessionLocal()
    session_uuid = str(uuid4())
    date = pd.to_datetime("2025-01-01 00:00:00")

    save_model_run(
        symbol="BTCUSDT",
        interval="30",
        date=date,
        params={"a": 1, "b": 2},
        loss=0.05,
        pnl=15.5,
        total_trades=3,
        winrate=0.66,
        risk_pct=0.1,
        retrained=True,
        triggered_restart=False,
        session_uuid=session_uuid,
        balance=1000.0,
        best_params=json.dumps({"a": 1, "b": 2})
    )


    result = session.query(ModelRun).filter_by(session_uuid=session_uuid).first()
    assert result is not None
    assert result.symbol == "BTCUSDT"
    assert result.interval == "30"
    assert result.total_trades == 3
    assert result.pnl == 15.5
    assert result.retrained is True
    assert result.triggered_restart is False
    assert result.session_uuid == session_uuid
    
    session.delete(result)
    session.commit()
    session.close()

def test_save_model_run_requires_session_uuid():
    with pytest.raises(ValueError):
        save_model_run(
            symbol="BTCUSDT",
            interval="30",
            date=pd.to_datetime("2025-01-01"),
            params={},
            loss=0.0,
            pnl=0.0,
            total_trades=0,
            winrate=0.0,
            risk_pct=0.01,
            retrained=False,
            triggered_restart=False,
            session_uuid=None
        )
