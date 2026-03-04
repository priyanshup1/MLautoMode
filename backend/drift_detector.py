import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def detect_data_drift(train_df: pd.DataFrame, new_df: pd.DataFrame, threshold: float = 0.1) -> dict:
    """
    Detects if the distribution of new data is significantly different from training data.
    Uses Kolmogorov-Smirnov (KS) test for numeric columns.
    Returns a dict with drift status and details per column.
    """
    drift_detected = False
    drift_details = []

    # Only process columns that exist in both DataFrames
    common_cols = list(set(train_df.columns).intersection(new_df.columns))

    for col in common_cols:
        # We only do KS test on numeric columns for simplicity
        if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(new_df[col]):
            train_series = train_df[col].dropna()
            new_series = new_df[col].dropna()

            if len(train_series) > 0 and len(new_series) > 0:
                stat, p_value = ks_2samp(train_series, new_series)
                
                # If p-value is small (e.g. < 0.05), the distributions are significantly different
                is_drifted = p_value < 0.05
                if is_drifted:
                    drift_detected = True

                drift_details.append({
                    "column": col,
                    "ks_statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 4),
                    "is_drifted": is_drifted
                })

    drift_details.sort(key=lambda x: x["p_value"]) # Sort by most drifted

    return {
        "drift_detected": drift_detected,
        "details": drift_details,
        "message": "Significant distribution shift detected. Retraining recommended." if drift_detected else "No significant data drift detected."
    }
