import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd

from strategy.walkforward import update_tracking

@patch("strategy.walkforward.save_model_run")
def test_update_tracking_calls_save_model_run(mock_save_model_run):
    test_date = pd.to_datetime("2023-01-01 00:00:00")
    df_test_prepared = pd.DataFrame(index=[test_date])
    df_test = pd.DataFrame(index=[test_date])
    interval = '30'

    config = {
        "symbol": "BTCUSDT",
        "risk_pct": 0.05,
        "balance": 1000,
        "trade_log": [],
        "best_params": {"long_score_threshold": 7},
        "days_elapsed": 0
    }

    result = {
        "winrate": 0.6,
        "pnl": 10,
        "total_trades": 5
    }

    update_tracking(config, interval, result, df_test, df_test_prepared)

    assert len(config["trade_log"]) == 1
    assert config["days_elapsed"] == 1
    mock_save_model_run.assert_called_once()

    args, kwargs = mock_save_model_run.call_args

    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["pnl"] == 10
    assert kwargs["winrate"] == 0.6
    assert kwargs["retrained"] is False  # потому что pnl > 0
    assert isinstance(kwargs["params"], dict)

