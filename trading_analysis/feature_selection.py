import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import pytest

def select_important_features(
    df,
    feature_cols,
    target_col,
    min_importance=0.01,
    random_state=42,
    verbose=False
):
    df_clean = df[feature_cols + [target_col]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError("Target must have at least two classes (got only one)")

    discrete_mask = [pd.api.types.is_integer_dtype(X[col]) for col in X.columns]

    # фиксируем random_state для воспроизводимости
    importances = mutual_info_classif(
        X,
        y,
        discrete_features=discrete_mask,
        random_state=random_state
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    important_features = importance_df[importance_df["importance"] >= min_importance]["feature"].tolist()

    if verbose:
        print(f"Selected {len(important_features)} features (min_importance={min_importance})")
        print(importance_df)

    return important_features, importance_df

def test_raises_on_single_class_target():
    df = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'target': [1] * 100  # только один класс
    })

    with pytest.raises(ValueError, match="Target must have at least two classes"):
        select_important_features(df, ['feature_1', 'feature_2'], 'target')

def test_returns_empty_when_no_important_features():
    df = pd.DataFrame({
        'f1': np.random.rand(100),
        'f2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

    # Поднимаем порог до почти невозможного
    selected, _ = select_important_features(df, ['f1', 'f2'], 'target', min_importance=1.0)

    assert selected == [], "Если нет признаков с достаточной важностью, должен быть пустой список"

def test_select_important_features_verbose_capfd():
    df = pd.DataFrame({
        'f1': np.random.rand(100),
        'f2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

    selected, _ = select_important_features(df, ['f1', 'f2'], 'target', verbose=True)
    assert isinstance(selected, list)

def test_random_state_makes_result_deterministic():
    df = pd.DataFrame({
        'f1': np.random.rand(100),
        'f2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

    s1, _ = select_important_features(df, ['f1', 'f2'], 'target', random_state=42)
    s2, _ = select_important_features(df, ['f1', 'f2'], 'target', random_state=42)

    assert s1 == s2, "Результат должен быть детерминированным при одинаковом random_state"
