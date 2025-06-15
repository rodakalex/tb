import pandas as pd
import numpy as np
from trading_analysis.feature_selection import select_important_features

def test_select_important_features_basic():
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100) * 2,
        'feature_3': np.random.rand(100) - 0.5,
        'target': np.random.randint(0, 2, 100)
    })

    selected, importance_df = select_important_features(df, feature_cols=['feature_1', 'feature_2', 'feature_3'], target_col='target')

    assert isinstance(selected, list), "Функция должна возвращать список признаков в первом элементе"
    assert len(selected) > 0, "Список отобранных признаков не должен быть пустым"
    assert set(selected).issubset({'feature_1', 'feature_2', 'feature_3'})


def test_select_important_features_with_min_importance():
    df = pd.DataFrame({
        'f1': np.random.rand(100),
        'f2': np.random.rand(100),
        'f3': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

    selected, _ = select_important_features(df, feature_cols=['f1', 'f2', 'f3'], target_col='target', min_importance=0.1)

    assert all(isinstance(f, str) for f in selected), "Все отобранные признаки должны быть строками"
    assert len(selected) <= 3
