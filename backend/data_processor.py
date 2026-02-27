import pandas as pd
import numpy as np
import io


def load_dataset(file_content: bytes) -> pd.DataFrame:
    try:
        # Decode and strip BOM
        text = file_content.decode('utf-8-sig').strip()
        df = pd.read_csv(io.StringIO(text))
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {str(e)}")


def analyze_dataset(df: pd.DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {available}"
        )

    num_rows, num_cols = df.shape
    missing_values = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    target_data = df[target_column]

    if pd.api.types.is_numeric_dtype(target_data):
        unique_values = target_data.nunique()
        problem_type = "Regression" if unique_values > 20 else "Classification"
    else:
        problem_type = "Classification"

    summary = {
        "problem_type": problem_type,
        "rows": num_rows,
        "columns": num_cols,
        "column_names": list(df.columns),
        "missing_values": missing_values,
        "data_types": data_types,
    }

    if problem_type == "Classification":
        class_dist = target_data.value_counts(normalize=True).to_dict()
        summary["class_distribution"] = {str(k): float(v) for k, v in class_dist.items()}

    return summary


def get_full_statistics(df: pd.DataFrame, target_column: str) -> dict:
    """Per-column descriptive statistics."""
    stats = []
    for col in df.columns:
        col_data = df[col]
        entry = {
            "column": col,
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isnull().sum()),
            "missing_pct": round(col_data.isnull().mean() * 100, 2),
            "unique_count": int(col_data.nunique()),
            "is_target": col == target_column,
        }
        if pd.api.types.is_numeric_dtype(col_data):
            entry.update({
                "mean": round(float(col_data.mean()), 4) if not col_data.isnull().all() else None,
                "std": round(float(col_data.std()), 4) if not col_data.isnull().all() else None,
                "min": round(float(col_data.min()), 4) if not col_data.isnull().all() else None,
                "max": round(float(col_data.max()), 4) if not col_data.isnull().all() else None,
                "median": round(float(col_data.median()), 4) if not col_data.isnull().all() else None,
                "skewness": round(float(col_data.skew()), 4) if not col_data.isnull().all() else None,
                "kurtosis": round(float(col_data.kurt()), 4) if not col_data.isnull().all() else None,
            })
        stats.append(entry)
    return {"columns": stats}


def domain_analysis(df: pd.DataFrame, target_column: str) -> dict:
    """Flag data quality issues."""
    flags = []
    for col in df.columns:
        col_data = df[col].dropna()
        issues = []

        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 50:
            issues.append(f"High missing data: {missing_pct:.1f}%")

        if col_data.nunique() <= 1:
            issues.append("Zero or constant variance — no predictive value")

        if pd.api.types.is_numeric_dtype(col_data) and len(col_data) > 0:
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = int(((col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)).sum())
            outlier_pct = round(outlier_count / len(col_data) * 100, 2)
            if outlier_pct > 5:
                issues.append(f"Outliers detected: {outlier_pct}% of values ({outlier_count} rows)")

            # Z-score extreme values check
            if col_data.std() > 0:
                z_extreme = int((np.abs((col_data - col_data.mean()) / col_data.std()) > 4).sum())
                if z_extreme > 0:
                    issues.append(f"Extreme values (|z|>4): {z_extreme} rows")

        if issues:
            flags.append({"column": col, "issues": issues, "severity": "High" if len(issues) >= 2 else "Medium"})

    return {"flags": flags, "total_flagged": len(flags)}


def correlation_analysis(df: pd.DataFrame, target_column: str) -> dict:
    """Correlation matrix + multicollinearity warnings."""
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        return {"target_correlations": [], "multicollinearity_warnings": []}

    corr_matrix = numeric_df.corr()

    # Target correlations
    target_corr = []
    if target_column in corr_matrix.columns:
        tc = corr_matrix[target_column].drop(target_column).dropna()
        target_corr = [
            {"feature": feat, "correlation": round(float(val), 4)}
            for feat, val in tc.sort_values(key=abs, ascending=False).items()
        ]

    # Multicollinearity between independent variables
    warnings = []
    feature_cols = [c for c in numeric_df.columns if c != target_column]
    seen = set()
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i + 1:]:
            if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                r = corr_matrix.loc[col1, col2]
                if abs(r) >= 0.85:
                    pair_key = tuple(sorted([col1, col2]))
                    if pair_key not in seen:
                        seen.add(pair_key)
                        warnings.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": round(float(r), 4),
                        })

    # Full matrix as list of lists (top 15 features max)
    top_feats = [c for c in feature_cols[:15]]
    if target_column in corr_matrix.columns:
        top_feats = top_feats[:14]
        top_feats.append(target_column)
    matrix_cols = [c for c in top_feats if c in corr_matrix.columns]
    matrix_data = {
        "columns": matrix_cols,
        "values": [[round(float(corr_matrix.loc[r, c]), 3) if r in corr_matrix.index and c in corr_matrix.columns else 0
                    for c in matrix_cols] for r in matrix_cols]
    }

    return {
        "target_correlations": target_corr,
        "multicollinearity_warnings": warnings,
        "matrix": matrix_data,
    }


def univariate_data(df: pd.DataFrame) -> dict:
    """Histogram bin data for numeric cols, value counts for categorical."""
    numeric_results = []
    categorical_results = []

    for col in df.columns:
        col_data = df[col].dropna()
        if pd.api.types.is_numeric_dtype(col_data) and len(col_data) > 0:
            counts, bin_edges = np.histogram(col_data, bins=min(20, col_data.nunique()))
            numeric_results.append({
                "column": col,
                "bins": [{"range": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}", "count": int(counts[i])}
                         for i in range(len(counts))]
            })
        elif col_data.dtype == object or str(col_data.dtype) == 'category':
            vc = col_data.value_counts().head(20)
            categorical_results.append({
                "column": col,
                "values": [{"label": str(k), "count": int(v)} for k, v in vc.items()]
            })

    return {"numeric": numeric_results, "categorical": categorical_results}


def bivariate_data(df: pd.DataFrame, target_column: str, problem_type: str) -> dict:
    """Feature vs target scatter/box data (sampled)."""
    sample_df = df.sample(min(500, len(df)), random_state=42)
    results = []
    feature_cols = [c for c in df.columns if c != target_column][:8]  # limit to 8

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            pairs = sample_df[[col, target_column]].dropna()
            results.append({
                "feature": col,
                "type": "scatter",
                "data": [{"x": round(float(r[col]), 4), "y": round(float(r[target_column]), 4) if pd.api.types.is_numeric_dtype(df[target_column]) else str(r[target_column])}
                         for _, r in pairs.iterrows()]
            })

    return {"features": results}
