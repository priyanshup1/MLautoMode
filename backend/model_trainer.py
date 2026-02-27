import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, roc_auc_score
)


def _get_feature_importance(model, X_train):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        coef = model.coef_
        return np.abs(coef[0] if coef.ndim > 1 else coef).tolist()
    return None


def train_classification_models(X_train, y_train, X_test, y_test, feature_names, selected_models=None):
    default_models = ["Logistic Regression", "Random Forest", "SVM", "KNN"]
    if selected_models is None:
        selected_models = default_models

    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
    }

    unique_classes = np.unique(y_train)
    is_binary = len(unique_classes) == 2

    results = []
    for name in selected_models:
        if name not in models_def:
            continue
        model = models_def[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        roc_data = None
        auc_score = None
        if is_binary and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = float(roc_auc_score(y_test, y_prob))
            # sample the curve to 50 pts
            step = max(1, len(fpr) // 50)
            roc_data = {
                "fpr": [round(float(v), 4) for v in fpr[::step]],
                "tpr": [round(float(v), 4) for v in tpr[::step]],
                "auc": round(auc_score, 4),
            }

        results.append({
            "model_name": name,
            "metrics": {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc_score,
            },
            "confusion_matrix": cm,
            "roc_curve": roc_data,
            "feature_importance": _get_feature_importance(model, X_train),
        })

    results.sort(key=lambda x: x["metrics"]["Accuracy"], reverse=True)
    return results


def train_regression_models(X_train, y_train, X_test, y_test, feature_names, selected_models=None):
    default_models = ["Linear Regression", "Random Forest Regressor", "Decision Tree", "SVR"]
    if selected_models is None:
        selected_models = default_models

    models_def = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
    }

    results = []
    for name in selected_models:
        if name not in models_def:
            continue
        model = models_def[name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        # Sample actual vs predicted (max 200 rows)
        sample_size = min(200, len(y_test))
        idx = np.random.choice(len(y_test), sample_size, replace=False)
        y_test_arr = np.array(y_test)
        actual_vs_pred = [
            {"actual": round(float(y_test_arr[i]), 4), "predicted": round(float(y_pred[i]), 4)}
            for i in sorted(idx)
        ]
        residuals = [
            {"index": int(i), "residual": round(float(y_test_arr[i] - y_pred[i]), 4)}
            for i in sorted(idx)
        ]

        results.append({
            "model_name": name,
            "metrics": {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2,
            },
            "actual_vs_predicted": actual_vs_pred,
            "residuals": residuals,
            "feature_importance": _get_feature_importance(model, X_train),
        })

    results.sort(key=lambda x: x["metrics"]["R2 Score"], reverse=True)
    return results


def train_and_evaluate(df: pd.DataFrame, target_column: str, problem_type: str,
                       models_to_train: list = None) -> dict:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    feature_names = X.columns.tolist()

    stratify = y if problem_type == "Classification" and y.value_counts().min() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    if problem_type == "Classification":
        results = train_classification_models(X_train, y_train, X_test, y_test, feature_names, models_to_train)
    else:
        results = train_regression_models(X_train, y_train, X_test, y_test, feature_names, models_to_train)

    return {
        "results": results,
        "feature_names": feature_names,
        "best_model": results[0]["model_name"] if results else None,
    }
