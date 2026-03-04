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
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
import optuna
from sklearn.model_selection import cross_val_score
import time
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_feature_importance(model, feature_names):
    importance_list = []
    if hasattr(model, "feature_importances_"):
        importance_list = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importance_list = np.abs(coef[0] if coef.ndim > 1 else coef).tolist()
    
    if importance_list and feature_names and len(importance_list) == len(feature_names):
        feature_importance = [{"feature": name, "importance": round(float(imp), 4)} 
                              for name, imp in zip(feature_names, importance_list)]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        return feature_importance
    return None


def _generate_shap_plot(model, X_train):
    try:
        if type(model).__name__ in ["RandomForestClassifier", "RandomForestRegressor", "DecisionTreeRegressor"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        elif type(model).__name__ in ["LogisticRegression", "LinearRegression"]:
            # Linear explainer requires training data distribution
            masker = shap.maskers.Independent(data=X_train)
            explainer = shap.LinearExplainer(model, masker=masker)
            shap_values = explainer.shap_values(X_train)
        elif type(model).__name__ in ["XGBClassifier", "XGBRegressor"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        else:
            return None # KernelExplainer is too slow for real-time generic use
            
        plt.figure(figsize=(8, 6))
        
        # Handle multiclass shape difference in shap
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap.summary_plot(shap_values[1], X_train, show=False) # Plot for class 1
        else:
            shap.summary_plot(shap_values, X_train, show=False)
            
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"SHAP error: {e}")
        return None

def train_classification_models(X_train, y_train, X_test, y_test, feature_names, selected_models=None, pro_mode=False):
    default_models = ["Logistic Regression", "Random Forest", "SVM", "KNN"]
    if selected_models is None:
        selected_models = default_models

    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
    }
    if XGB_AVAILABLE:
        models_def["XGBoost"] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
        if "XGBoost" not in default_models:
             default_models.insert(0, "XGBoost")

    unique_classes = np.unique(y_train)
    is_binary = len(unique_classes) == 2

    results = []
    for name in selected_models:
        if name not in models_def:
            continue
            
        start_time = time.time()
        best_params = None
        cv_score = None
        model = models_def[name]

        if pro_mode:
            # Optuna hyperparameter tuning
            def objective(trial):
                if name == "Random Forest":
                    n_estimators = trial.suggest_int("n_estimators", 50, 200)
                    max_depth = trial.suggest_int("max_depth", 3, 15)
                    trial_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif name == "SVM":
                    C = trial.suggest_float("C", 0.1, 10.0, log=True)
                    trial_model = SVC(C=C, probability=True, random_state=42)
                elif name == "Logistic Regression":
                    C = trial.suggest_float("C", 0.1, 10.0, log=True)
                    trial_model = LogisticRegression(C=C, max_iter=1000, random_state=42)
                elif name == "KNN":
                    n_neighbors = trial.suggest_int("n_neighbors", 3, 11)
                    trial_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                elif name == "XGBoost":
                    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                    max_depth = trial.suggest_int("max_depth", 3, 10)
                    trial_model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=42, use_label_encoder=False, eval_metric="logloss")
                else:
                    trial_model = models_def[name]
                
                score = cross_val_score(trial_model, X_train, y_train, cv=3, scoring="accuracy").mean()
                return score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=5) # 5 trials for speed
            
            best_params = study.best_params
            cv_score = float(study.best_value)
            
            # Reconstruct and train best model
            if name == "Random Forest":
                model = RandomForestClassifier(**best_params, random_state=42)
            elif name == "SVM":
                model = SVC(**best_params, probability=True, random_state=42)
            elif name == "Logistic Regression":
                model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
            elif name == "KNN":
                model = KNeighborsClassifier(**best_params)
            elif name == "XGBoost":
                model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric="logloss")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        # Enhanced Confusion Matrix for Heatmap
        raw_cm = confusion_matrix(y_test, y_pred)
        classes = sorted(list(unique_classes))
        cm_data = {
            "matrix": raw_cm.tolist(),
            "labels": [str(c) for c in classes]
        }

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

        train_time = time.time() - start_time
        
        shap_b64 = _generate_shap_plot(model, X_train) if pro_mode else None

        results.append({
            "model_name": name,
            "metrics": {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc_score,
            },
            "train_time_sec": round(train_time, 2),
            "cv_score": round(cv_score, 4) if cv_score else None,
            "best_params": best_params,
            "confusion_matrix": cm_data,
            "roc_curve": roc_data,
            "feature_importance": _get_feature_importance(model, feature_names),
            "shap_plot": shap_b64,
            "model": model, # Return the instance for persistence
        })

    results.sort(key=lambda x: x["metrics"]["Accuracy"], reverse=True)
    return results


def train_regression_models(X_train, y_train, X_test, y_test, feature_names, selected_models=None, pro_mode=False):
    default_models = ["Linear Regression", "Random Forest Regressor", "Decision Tree", "SVR"]
    if selected_models is None:
        selected_models = default_models

    models_def = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(),
    }
    if XGB_AVAILABLE:
        models_def["XGBoost Regressor"] = XGBRegressor(random_state=42)
        if "XGBoost Regressor" not in default_models:
            default_models.insert(0, "XGBoost Regressor")

    results = []
    for name in selected_models:
        if name not in models_def:
            continue
            
        start_time = time.time()
        best_params = None
        cv_score = None
        model = models_def[name]

        if pro_mode:
            # Optuna hyperparameter tuning
            def objective(trial):
                if name == "Random Forest Regressor":
                    n_estimators = trial.suggest_int("n_estimators", 50, 200)
                    max_depth = trial.suggest_int("max_depth", 3, 15)
                    trial_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                elif name == "SVR":
                    C = trial.suggest_float("C", 0.1, 10.0, log=True)
                    trial_model = SVR(C=C)
                elif name == "Decision Tree":
                    max_depth = trial.suggest_int("max_depth", 3, 15)
                    trial_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                elif name == "XGBoost Regressor":
                    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                    max_depth = trial.suggest_int("max_depth", 3, 10)
                    trial_model = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                else:
                    trial_model = models_def[name]
                
                score = cross_val_score(trial_model, X_train, y_train, cv=3, scoring="r2").mean()
                return score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=5) # 5 trials for speed
            
            best_params = study.best_params
            cv_score = float(study.best_value)
            
            # Reconstruct and train best model
            if name == "Random Forest Regressor":
                model = RandomForestRegressor(**best_params, random_state=42)
            elif name == "SVR":
                model = SVR(**best_params)
            elif name == "Decision Tree":
                model = DecisionTreeRegressor(**best_params, random_state=42)
            elif name == "XGBoost Regressor":
                model = XGBRegressor(**best_params, random_state=42)

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

        train_time = time.time() - start_time

        shap_b64 = _generate_shap_plot(model, X_train) if pro_mode else None

        results.append({
            "model_name": name,
            "metrics": {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2,
            },
            "train_time_sec": round(train_time, 2),
            "cv_score": round(cv_score, 4) if cv_score else None,
            "best_params": best_params,
            "actual_vs_predicted": actual_vs_pred,
            "residuals": residuals,
            "feature_importance": _get_feature_importance(model, feature_names),
            "shap_plot": shap_b64,
            "model": model, # Return the instance for persistence
        })

    results.sort(key=lambda x: x["metrics"]["R2 Score"], reverse=True)
    return results


def train_and_evaluate(df: pd.DataFrame, target_column: str, problem_type: str,
                       models_to_train: list = None, pro_mode: bool = False) -> dict:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    feature_names = X.columns.tolist()

    stratify = y if problem_type == "Classification" and y.value_counts().min() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    if problem_type == "Classification":
        results = train_classification_models(X_train, y_train, X_test, y_test, feature_names, models_to_train, pro_mode)
    else:
        results = train_regression_models(X_train, y_train, X_test, y_test, feature_names, models_to_train, pro_mode)

    return {
        "results": results,
        "feature_names": feature_names,
        "best_model": results[0]["model_name"] if results else None,
    }
