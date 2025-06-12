from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def select_important_features(df, feature_cols, target_col, min_importance=0.01):
    df_clean = df[feature_cols + [target_col]].dropna()
    X = df_clean[feature_cols].astype(float)
    y = df_clean[target_col].astype(int)

    X_scaled = MinMaxScaler().fit_transform(X)
    importances = mutual_info_classif(X_scaled, y, discrete_features=True)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    important_features = importance_df.query("importance >= @min_importance")["feature"].tolist()
    return important_features, importance_df
