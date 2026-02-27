import pandas as pd
import numpy as np


def generate_visualizations(df: pd.DataFrame, target_column: str, problem_type: str) -> dict:
    charts = {}

    # 1. Target Distribution
    try:
        if problem_type == "Classification":
            dist = df[target_column].value_counts().to_dict()
            dist_data = [{"name": str(k), "value": int(v)} for k, v in dist.items()]
            charts["target_distribution"] = {
                "type": "bar",
                "data": dist_data,
                "explanation": (
                    "Target distribution shows the balance of classes. "
                    "Imbalanced classes (one class >> others) may require resampling techniques "
                    "like SMOTE or class-weight adjustments for reliable predictions."
                ),
            }
        else:
            counts, bins = np.histogram(df[target_column].dropna(), bins=12)
            dist_data = [
                {"name": f"{bins[i]:.2f}–{bins[i+1]:.2f}", "value": int(counts[i])}
                for i in range(len(counts))
            ]
            charts["target_distribution"] = {
                "type": "bar",
                "data": dist_data,
                "explanation": (
                    "The target distribution reveals the spread of your numeric outcome. "
                    "A skewed distribution may require log-transformation for better model performance."
                ),
            }
    except Exception as e:
        charts["target_distribution"] = {"error": str(e)}

    # 2. Feature Correlation with Target (top 12)
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1 and target_column in numeric_df.columns:
            corr = numeric_df.corr()
            target_corr = corr[target_column].drop(target_column).dropna()
            top_feats = target_corr.abs().nlargest(12).index
            heatmap_data = [
                {"name": feat, "correlation": round(float(target_corr[feat]), 3)}
                for feat in top_feats
            ]
            charts["correlation"] = {
                "type": "bar",
                "data": heatmap_data,
                "explanation": (
                    "Correlation strength shows each feature's linear relationship with the target. "
                    "Positive bars indicate features that increase together with the target; "
                    "negative bars show inverse relationships. Features near zero may have little predictive value."
                ),
            }
    except Exception as e:
        charts["correlation"] = {"error": str(e)}

    # 3. Box plot outlier data per top numeric feature
    try:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != target_column][:6]
        box_data = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            q1, q3 = float(col_data.quantile(0.25)), float(col_data.quantile(0.75))
            box_data.append({
                "column": col,
                "min": round(float(col_data.min()), 4),
                "q1": round(q1, 4),
                "median": round(float(col_data.median()), 4),
                "q3": round(q3, 4),
                "max": round(float(col_data.max()), 4),
                "iqr": round(q3 - q1, 4),
            })
        charts["box_plot"] = {
            "data": box_data,
            "explanation": (
                "Box plots reveal the spread and outliers in each feature. "
                "The box spans Q1–Q3 (interquartile range). Points beyond 1.5×IQR are outliers "
                "and may distort model training if not handled."
            ),
        }
    except Exception as e:
        charts["box_plot"] = {"error": str(e)}

    return charts
