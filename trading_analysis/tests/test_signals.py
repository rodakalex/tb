import numpy as np
import pandas as pd
import pytest
from trading_analysis.signals import generate_signals

def get_minimal_test_df(n=20):
    data = {
        'close': list(range(100, 100 + n)),
        'high': list(range(101, 101 + n)),
        'low': list(range(99, 99 + n)),
        'open': list(range(99, 99 + n)),
        'volume': [1000 + i*10 for i in range(n)],
        'EMA_9': [100 + i*0.5 for i in range(n)],
        'EMA_21': [99 + i*0.4 for i in range(n)],
        'EMA_200': [95 for _ in range(n)],
        'MACD': [0.1*i for i in range(n)],
        'Signal': [0.1*i - 0.05 for i in range(n)],
        'BBM': [100 for _ in range(n)],
        'BBP': [0.6 for _ in range(n)],
        'Volume_SMA_20': [1000 for _ in range(n)],
        'RSI': [50 + i for i in range(n)],
    }
    return pd.DataFrame(data)

def test_generate_signals_basic_long():
    df = get_minimal_test_df()
    result = generate_signals(df)

    assert "long_score" in result.columns
    assert "long_entry" in result.columns
    assert result["long_entry"].iloc[-1] in [True, False]

def test_macd_signal_generation():
    df = get_minimal_test_df()
    df["MACD"] = [0.2, 0.3, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5,
                  2.7, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
    df["Signal"] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    df["EMA_200"] = np.full(len(df), 90)

    result = generate_signals(df)
    assert (result["long_macd"] == 1).sum() > 0


def test_empty_df():
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        generate_signals(df)

def test_threshold_param_affects_entry():
    df = get_minimal_test_df()
    params = {
        "long_score_threshold": 100
    }
    result = generate_signals(df, params=params)
    assert result["long_entry"].sum() == 0

def test_stochrsi_none(monkeypatch):
    def fake_stochrsi(*args, **kwargs):
        return None

    import pandas_ta as ta
    monkeypatch.setattr(ta, "stochrsi", fake_stochrsi)

    df = get_minimal_test_df()
    result = generate_signals(df)
    assert "StochRSI_K" in result.columns
    assert result["StochRSI_K"].isna().all()

def test_custom_long_signals_only_volume():
    df = get_minimal_test_df()
    params = {
        "enabled_long_signals": ["volume_above_avg"],
        "long_score_threshold": 1,
        "w_volume": 1
    }
    result = generate_signals(df, params=params)

    assert set(result["long_score"].unique()).issubset({0, 1})
    assert (result["long_score"] == result["volume_above_avg"]).all()

def test_generate_signals_short_entry():
    df = get_minimal_test_df()
    df["EMA_200"] = df["EMA_200"] + np.linspace(10, -10, len(df))  # сделать downtrend
    result = generate_signals(df)
    assert "short_entry" in result.columns
    assert result["short_entry"].iloc[-1] in [True, False]

def test_trend_strength_filter_blocks_entry():
    df = get_minimal_test_df()
    df["ADX"] = 10  # недостаточно для trend_strength
    result = generate_signals(df)
    assert result["long_entry"].sum() == 0

def test_short_entry_from_roc_only():
    df = get_minimal_test_df(n=80)
    df["ROC"] = -2
    df["EMA_200"] = np.linspace(120, 100, len(df))
    df["close"] = np.linspace(100, 120, len(df))
    df["high"] = df["close"] + 6
    df["low"] = df["close"] - 6
    df["volume"] = 1000

    params = {
        "enabled_short_signals": ["short_roc"],
        "short_score_threshold": 1,
        "use_atr_filter": False,
        "use_ema200_down_filter": False,
        "use_custom_roc": True
    }

    result = generate_signals(df, params)
    assert (result["short_score"] == result["short_roc"]).all()
    assert result["short_entry"].sum() > 0

