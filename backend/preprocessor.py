import pandas as pd
import numpy as np
import base64
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


def preprocess_data(df: pd.DataFrame, target_column: str, scale_features: bool = True,
                    scaler_type: str = "standard") -> dict:
    df_clean = df.copy()
    step_log = []

    stats = {
        "original_rows": len(df),
        "removed_duplicates": 0,
        "imputed_columns": [],
    }

    # 1. Remove duplicates
    before_len = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    after_len = len(df_clean)
    stats["removed_duplicates"] = before_len - after_len
    if stats["removed_duplicates"] > 0:
        step_log.append({"step": "Remove Duplicates",
                         "detail": f"Dropped {stats['removed_duplicates']} duplicate rows."})

    # Separate features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # 2. Impute missing values
    for col in X.columns:
        null_count = X[col].isnull().sum()
        if null_count > 0:
            stats["imputed_columns"].append(col)
            if pd.api.types.is_numeric_dtype(X[col]):
                fill_val = X[col].median()
                X[col] = X[col].fillna(fill_val)
                step_log.append({"step": "Impute (Numeric)",
                                  "detail": f"'{col}': filled {null_count} nulls with median ({fill_val:.4f})"})
            else:
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                step_log.append({"step": "Impute (Categorical)",
                                  "detail": f"'{col}': filled {null_count} nulls with mode ('{mode_val}')"})

    # 3. Encode categoricals
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        X = X.astype({col: int for col in X.select_dtypes(include="bool").columns})
        step_log.append({"step": "Encode Categoricals",
                          "detail": f"One-hot encoded: {categorical_cols}"})

    # Encode target if categorical
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_column)
        step_log.append({"step": "Encode Target",
                          "detail": f"Label encoded target '{target_column}'"})
    else:
        if y.isnull().any():
            y = y.fillna(y.median())

    # 4. Feature Scaling
    if scale_features:
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        scaler = scaler_map.get(scaler_type, StandardScaler())
        scaler_name = {"standard": "StandardScaler (Z-score)", "minmax": "MinMaxScaler",
                       "robust": "RobustScaler"}.get(scaler_type, "StandardScaler")
        col_names = X.columns
        X = pd.DataFrame(scaler.fit_transform(X), columns=col_names)
        step_log.append({"step": "Feature Scaling",
                          "detail": f"Applied {scaler_name} to all numeric features."})

    df_final = pd.concat([X, y], axis=1)

    # 5. Encode cleaned CSV as base64 for download
    csv_bytes = df_final.to_csv(index=False).encode("utf-8")
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")

    return {
        "cleaned_df": df_final,
        "stats": stats,
        "step_log": step_log,
        "cleaned_csv_b64": csv_b64,
    }
