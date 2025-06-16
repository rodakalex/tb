from trading_analysis.utils import prepare_params_for_logging

def test_prepare_params_for_logging_injects_signals():
    raw_params = {
        "long_score_threshold": 5,
        "risk_pct": 0.1,
        "w_macd": 1.0
    }
    config = {
        "best_params": {
            "enabled_long_signals": ["macd", "volume_above_avg"],
            "enabled_short_signals": ["short_rsi"]
        }
    }

    result = prepare_params_for_logging(raw_params, config)

    assert "enabled_long_signals" in result
    assert "enabled_short_signals" in result
    assert result["enabled_long_signals"] == ["macd", "volume_above_avg"]
    assert result["enabled_short_signals"] == ["short_rsi"]

def test_prepare_params_for_logging_handles_none():
    config = {}
    result = prepare_params_for_logging(None, config)
    assert result == {}
