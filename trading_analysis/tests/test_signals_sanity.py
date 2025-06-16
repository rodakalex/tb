from trading_analysis.signals import generate_signals
import pandas as pd
import numpy as np

from trading_analysis.indicators import calculate_indicators
from trading_analysis.signals import generate_signals

def test_signals_have_at_least_some_entries():
    df = pd.DataFrame({
        'timestamp': pd.date_range("2024-01-01", periods=200, freq="30min"),
        'open': np.random.rand(200) * 100,
        'high': np.random.rand(200) * 100 + 10,
        'low': np.random.rand(200) * 100,
        'close': np.random.rand(200) * 100,
        'volume': np.random.randint(1000, 10000, 200)
    }).set_index("timestamp")

    df = calculate_indicators(df)
    signals = generate_signals(df, {
        "long_score_threshold": 1,
        "short_score_threshold": 1,
        "use_atr_filter": False,
        "use_trend_filter": False,
        "use_ema200_down_filter": False,
    })

    assert signals["long_entry"].sum() > 0 or signals["short_entry"].sum() > 0, "Нет сигналов"
