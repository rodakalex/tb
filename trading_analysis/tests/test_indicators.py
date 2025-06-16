import pandas_ta as ta
import pandas as pd
import numpy as np
from trading_analysis.indicators import calculate_indicators

def test_calculate_indicators_outputs_valid_data():
    df = pd.DataFrame({
        'timestamp': pd.date_range("2024-01-01", periods=300, freq="30min"),
        'open': np.random.rand(300) * 100,
        'high': np.random.rand(300) * 100 + 100,
        'low': np.random.rand(300) * 100,
        'close': np.random.rand(300) * 100,
        'volume': np.random.randint(1000, 10000, 300)
    }).set_index("timestamp")

    result = calculate_indicators(df)

    # Логируем количество NaN в последних 30 свечах
    recent = result.iloc[-30:]
    nan_stats = recent.isnull().sum()
    print("NaN count in last 30 rows:")
    print(nan_stats[nan_stats > 0])

    assert not recent.isnull().any().any(), "NaN в индикаторах на последних свечах"

    rsi_clean = result['RSI'].dropna()
    print(f"RSI min: {rsi_clean.min()}, max: {rsi_clean.max()}")
    assert (rsi_clean >= 0).all() and (rsi_clean <= 100).all(), "RSI out of bounds"

    assert result['EMA_9'].var() > 0, "EMA_9 не меняется"
