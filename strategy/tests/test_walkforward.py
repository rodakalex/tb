from unittest.mock import patch
import pandas as pd
from uuid import uuid4

from strategy.walkforward import should_trigger_restart, update_tracking

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
        "days_elapsed": 0,
        "session_uuid": str(uuid4())
    }

    result = {
        "winrate": 0.6,
        "pnl": 10,
        "total_trades": 5
    }

    update_tracking(config, interval, result, df_test, df_test_prepared, triggered_restart=False)

    assert len(config["trade_log"]) == 1
    assert config["days_elapsed"] == 1
    mock_save_model_run.assert_called_once()

    _, kwargs = mock_save_model_run.call_args

    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["pnl"] == 10
    assert kwargs["winrate"] == 0.6
    assert kwargs["retrained"] is False
    assert kwargs["balance"] == 1000
    assert kwargs["best_params"] == {"long_score_threshold": 7}
    assert isinstance(kwargs["params"], dict)
    assert "long_score_threshold" in kwargs["params"]



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

    update_tracking(config, interval, result, df_test, df_test_prepared, False)

    assert len(config["trade_log"]) == 1
    assert config["days_elapsed"] == 1
    mock_save_model_run.assert_called_once()

    args, kwargs = mock_save_model_run.call_args

    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["pnl"] == 10
    assert kwargs["winrate"] == 0.6
    assert kwargs["retrained"] is False  # потому что pnl > 0
    assert isinstance(kwargs["params"], dict)

# Базовые данные, легко расширяемые
BASE_RESULT = {"final_balance": 1000, "max_loss_streak": 0, "total_trades": 1, "winrate": 1.0}
BASE_CONFIG = {"initial_balance": 1000}

def make(result_diff=None, config_diff=None):
    result = BASE_RESULT.copy()
    config = BASE_CONFIG.copy()
    if result_diff:
        result.update(result_diff)
    if config_diff:
        config.update(config_diff)
    return result, config

def test_drawdown_trigger():
    result, config = make({"final_balance": 890})
    assert should_trigger_restart(result, config) is True

def test_max_loss_streak_trigger():
    result, config = make({"max_loss_streak": 3})
    assert should_trigger_restart(result, config) is True

def test_no_trades_once_no_trigger():
    result, config = make({"total_trades": 0})
    assert should_trigger_restart(result, config) is False
    assert config["no_trade_windows"] == 1

def test_no_trades_twice_trigger():
    # эмулируем уже 1 раз без сделок
    result, config = make({"total_trades": 0}, {"no_trade_windows": 1})
    assert should_trigger_restart(result, config) is True

def test_winrate_drop_trigger():
    result, config = make({"winrate": 0.3})
    assert should_trigger_restart(result, config) is True

def test_no_trigger_if_all_good():
    result, config = make()
    assert should_trigger_restart(result, config) is False

def test_multiple_restart_conditions_triggered():
    result = {"final_balance": 850, "max_loss_streak": 4, "total_trades": 0, "winrate": 0.2}
    config = {"initial_balance": 1000, "no_trade_windows": 1}
    assert should_trigger_restart(result, config) is True
