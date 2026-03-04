import pandas as pd
import numpy as np
import base64
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

def engineer_features(df: pd.DataFrame, 
                      target_column: str, 
                      problem_type: str,
                      imputation_strategy: str = "median",
                      encoding_strategy: str = "onehot",
                      scale_features: bool = True,
                      scaler_type: str = "standard",
                      polynomial_degree: int = 1,
                      feature_selection_k: int = 0,
                      handle_outliers: bool = False) -> dict:
    """
    Advanced feature engineering pipeline.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target variable
        problem_type: "Classification" or "Regression"
        imputation_strategy: "mean", "median", "most_frequent", "constant"
        encoding_strategy: "onehot" (target encoding planned for later)
        scale_features: Whether to scale numeric features
        scaler_type: "standard", "minmax", "robust"
        polynomial_degree: Degree of polynomial features (1 = none)
        feature_selection_k: Number of top features to keep (0 = all)
        handle_outliers: Whether to remove outliers using Isolation Forest
    """
    df_clean = df.copy()
    step_log = []
    
    stats = {
        "original_rows": len(df),
        "removed_duplicates": 0,
        "original_features": len(df.columns) - 1,
        "final_features": 0
    }

    # 1. Remove duplicates
    before_len = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    after_len = len(df_clean)
    stats["removed_duplicates"] = before_len - after_len
    if stats["removed_duplicates"] > 0:
        step_log.append({"step": "Remove Duplicates", 
                         "detail": f"Dropped {stats['removed_duplicates']} duplicate rows."})

    # 1b. Outlier Removal (Pro Mode)
    if handle_outliers:
        numeric_only = df_clean.select_dtypes(include=[np.number])
        if not numeric_only.empty:
            iso = IsolationForest(contamination=0.05, random_state=42)
            outliers = iso.fit_predict(numeric_only)
            df_clean = df_clean[outliers == 1]
            removed = before_len - stats["removed_duplicates"] - len(df_clean)
            if removed > 0:
                step_log.append({"step": "Outlier Removal", 
                                 "detail": f"Removed {removed} rows using Isolation Forest (5% contamination)."})

    # Separate X and y
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # 2. Imputation
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy=imputation_strategy)
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
        step_log.append({"step": "Numeric Imputation", 
                         "detail": f"Applied '{imputation_strategy}' strategy to {len(numeric_cols)} columns."})

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        step_log.append({"step": "Categorical Imputation", 
                         "detail": "Applied 'most_frequent' strategy to categorical columns."})

    # 3. Encoding
    if categorical_cols:
        if encoding_strategy == "onehot":
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            # Convert bool to int
            bool_cols = X.select_dtypes(include=["bool"]).columns
            X[bool_cols] = X[bool_cols].astype(int)
            step_log.append({"step": "Encoding", 
                             "detail": f"One-hot encoded {len(categorical_cols)} categorical columns."})
        elif encoding_strategy == "target":
            # Simple Target Encoding: replace category with mean of target (regression) or mode/prob (classification)
            # For simplicity, we use the mean of target for each category
            for col in categorical_cols:
                target_mean = y.groupby(X[col]).mean()
                X[col] = X[col].map(target_mean)
                # Fill NAs in case a category was in X but not y or vice versa (though separated from same DF here)
                X[col] = X[col].fillna(y.mean())
            step_log.append({"step": "Encoding", 
                             "detail": f"Applied Target Encoding to {len(categorical_cols)} columns."})

    # Target encoding if not numeric
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_column)
        step_log.append({"step": "Target Encoding", 
                         "detail": f"Label encoded target '{target_column}'"})
    else:
        if y.isnull().any():
            y = y.fillna(y.median())

    # 4. Polynomial Features (Interaction terms)
    if polynomial_degree > 1:
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        poly_cols = X.columns # Use all columns for interaction if they are now numeric
        X_poly = poly.fit_transform(X)
        X = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(poly_cols))
        step_log.append({"step": "Polynomial Features", 
                         "detail": f"Generated degree {polynomial_degree} features (Total: {X.shape[1]})."})

    # 5. Feature Scaling
    if scale_features:
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        scaler = scaler_map.get(scaler_type, StandardScaler())
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        step_log.append({"step": "Scaling", 
                         "detail": f"Applied {scaler_type} scaling."})

    # 6. Feature Selection
    if feature_selection_k > 0 and feature_selection_k < len(X.columns):
        score_func = f_classif if problem_type == "Classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=feature_selection_k)
        X_selected = selector.fit_transform(X, y)
        selected_mask = selector.get_support()
        X = X.loc[:, selected_mask]
        step_log.append({"step": "Feature Selection", 
                         "detail": f"Selected top {feature_selection_k} features using {score_func.__name__}."})

    df_final = pd.concat([X, y], axis=1)
    stats["final_features"] = len(X.columns)

    # Encode for download
    csv_bytes = df_final.to_csv(index=False).encode("utf-8")
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")

    return {
        "cleaned_df": df_final,
        "stats": stats,
        "step_log": step_log,
        "cleaned_csv_b64": csv_b64,
    }
