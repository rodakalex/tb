import pandas as pd
from trading_analysis.feature_selection import select_important_features

def test_select_important_features_output_not_empty():
    df = pd.read_csv("tests/data/mock_signals.csv", index_col=0, parse_dates=True)
    features = [col for col in df.columns if col.startswith("long_") and "_entry" not in col]
    selected, importance_df = select_important_features(df, features, target_col="long_entry")

    assert isinstance(selected, list)
    assert all(isinstance(f, str) for f in selected)
    assert len(selected) > 0

def test_importance_sorted_desc():
    df = pd.read_csv("tests/data/mock_signals.csv", index_col=0, parse_dates=True)
    features = [col for col in df.columns if col.startswith("short_") and "_entry" not in col]
    _, importance_df = select_important_features(df, features, target_col="short_entry")

    importance_values = importance_df["importance"].tolist()
    assert importance_values == sorted(importance_values, reverse=True)
