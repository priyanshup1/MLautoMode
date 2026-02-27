"""
Deep Business Insight Generator
Generates structured 5-dimension insights: What, How, Why, Where, Impact
"""

DOMAIN_CONTEXTS = {
    # Classification keywords
    "churn": ("Customer Retention", "telecom / subscription businesses"),
    "fraud": ("Fraud Detection", "banking, e-commerce, insurance"),
    "disease": ("Healthcare Diagnostics", "hospitals, clinics, health-tech"),
    "cancer": ("Healthcare Diagnostics", "oncology centers, hospitals"),
    "diagnosis": ("Healthcare Diagnostics", "medical institutions"),
    "spam": ("Content Filtering", "email, social media platforms"),
    "default": ("Credit Risk", "banking, lending, fintech"),
    "churn": ("Customer Retention", "SaaS, telecom, subscription businesses"),
    "sentiment": ("Sentiment Analysis", "marketing, customer support"),
    # Regression keywords
    "price": ("Pricing Optimization", "real estate, e-commerce, retail"),
    "sales": ("Sales Forecasting", "retail, FMCG, supply chain"),
    "demand": ("Demand Planning", "logistics, inventory, manufacturing"),
    "salary": ("Compensation Planning", "HR tech, workforce analytics"),
    "revenue": ("Revenue Forecasting", "SaaS, finance departments"),
    "cost": ("Cost Prediction", "operations, procurement, manufacturing"),
    "temperature": ("Environmental Monitoring", "climate science, agriculture, IoT"),
    "stock": ("Financial Forecasting", "investment firms, trading desks"),
    "energy": ("Energy Optimization", "utilities, smart buildings"),
}


def _detect_domain(target_col: str) -> tuple:
    col_lower = target_col.lower()
    for keyword, (domain, context) in DOMAIN_CONTEXTS.items():
        if keyword in col_lower:
            return domain, context
    return "Predictive Analytics", "data-driven organizations across industries"


def generate_insight(problem_type: str, target_col: str, best_model: dict) -> dict:
    domain, context = _detect_domain(target_col)
    model_name = best_model["model_name"]
    metrics = best_model["metrics"]

    if problem_type == "Classification":
        primary_metric_name = "Accuracy"
        primary_metric_val = metrics.get("Accuracy", 0)
        primary_metric_str = f"{primary_metric_val * 100:.1f}%"
        auc = metrics.get("ROC-AUC")
        auc_str = f" (ROC-AUC: {auc:.3f})" if auc else ""
        f1 = metrics.get("F1 Score", 0)

        what = (
            f"The **{model_name}** model successfully learned to predict **'{target_col}'** "
            f"achieving **{primary_metric_str} accuracy**{auc_str} and an F1 score of {f1:.3f}."
        )
        how = (
            f"To operationalize this model: (1) Export and serve the model via an API endpoint. "
            f"(2) Pass new records through the same preprocessing pipeline. "
            f"(3) Use the predicted class label — and its confidence score — to trigger business workflows "
            f"(e.g., flag high-risk cases, personalize offers, or route support tickets automatically)."
        )
        why = _classification_why(model_name, primary_metric_val, f1, auc)
        where = (
            f"This model is most applicable in **{domain}** contexts — specifically in {context}. "
            f"Integration points include CRM systems, alert dashboards, batch scoring pipelines, "
            f"and real-time decision engines."
        )
        impact = _classification_impact(primary_metric_val, f1, target_col, domain)

    else:
        r2 = metrics.get("R2 Score", 0)
        rmse = metrics.get("RMSE", 0)
        mae = metrics.get("MAE", 0)

        what = (
            f"The **{model_name}** model predicts the numeric value of **'{target_col}'** "
            f"with an **R² score of {r2:.4f}**, explaining {r2 * 100:.1f}% of variance in the target. "
            f"Average prediction error (MAE): **{mae:.4f}** units."
        )
        how = (
            f"To deploy: (1) Integrate the model into your data pipeline. "
            f"(2) Feed preprocessed feature vectors at the point of decision. "
            f"(3) Use the predicted value to set budgets, prices, quotas, or alerts. "
            f"Confidence bands can be computed using prediction intervals."
        )
        why = _regression_why(model_name, r2, rmse)
        where = (
            f"This regression model is best suited for **{domain}** applications in {context}. "
            f"Typical use cases include forecasting dashboards, automated pricing engines, "
            f"resource allocation systems, and demand planning tools."
        )
        impact = _regression_impact(r2, rmse, mae, target_col, domain)

    return {
        "what": what,
        "how": how,
        "why": why,
        "where": where,
        "impact": impact,
        "domain": domain,
        "context": context,
        "model_name": model_name,
    }


def _classification_why(model_name, accuracy, f1, auc):
    reasons = []
    if model_name in ("Random Forest", "XGBoost"):
        reasons.append("ensemble learning captures complex non-linear patterns and feature interactions")
    elif model_name == "Logistic Regression":
        reasons.append("a linear decision boundary was sufficient, suggesting clear class separability")
    elif model_name == "SVM":
        reasons.append("SVMs excel at finding optimal margins, particularly in high-dimensional spaces")
    elif model_name == "KNN":
        reasons.append("locally similar data points share the same class, and distance-based voting works well here")

    perf_note = ""
    if accuracy >= 0.9:
        perf_note = "The dataset is well-structured with strong signal in the features."
    elif accuracy >= 0.75:
        perf_note = "Performance is solid; consider feature engineering or hyperparameter tuning for further gains."
    else:
        perf_note = "Performance suggests class overlap or noise — review feature quality and consider SMOTE for imbalance."

    return f"**{model_name}** outperformed others because {', '.join(reasons) if reasons else 'it suited the data structure best'}. {perf_note}"


def _regression_why(model_name, r2, rmse):
    reasons = []
    if model_name in ("Random Forest Regressor", "XGBoost Regressor"):
        reasons.append("ensemble trees capture non-linear feature interactions missed by linear models")
    elif model_name == "Linear Regression":
        reasons.append("the target has a strong linear relationship with the features")
    elif model_name == "Decision Tree":
        reasons.append("rule-based splits aligned well with the natural data boundaries")
    elif model_name == "SVR":
        reasons.append("SVR's epsilon-insensitive loss reduced sensitivity to noisy outliers")

    perf_note = ""
    if r2 >= 0.85:
        perf_note = "Excellent predictive power. The model is production-ready."
    elif r2 >= 0.6:
        perf_note = "Good fit. More feature engineering or advanced models could further reduce RMSE."
    else:
        perf_note = "Modest fit — consider adding domain-specific features or non-linear transformations."

    return f"**{model_name}** won because {', '.join(reasons) if reasons else 'it was best suited to your data distribution'}. {perf_note}"


def _classification_impact(accuracy, f1, target_col, domain):
    impact_lines = [
        f"**Efficiency**: Automating '{target_col}' prediction eliminates manual review effort, "
        f"scaling decision-making without adding headcount.",
    ]
    if accuracy >= 0.85:
        impact_lines.append(
            f"**Risk Reduction**: At {accuracy * 100:.1f}% accuracy, the model correctly classifies ~{int(accuracy * 1000)} "
            f"out of every 1,000 cases — significantly reducing errors compared to random guessing ({50}%)."
        )
    impact_lines.append(
        f"**Revenue / Cost**: In {domain}, even a 5–10% improvement in prediction accuracy typically "
        f"translates to measurable cost savings (fewer false positives) or revenue gains (better targeting)."
    )
    impact_lines.append(
        f"**Scalability**: Once deployed, the model scores new data in milliseconds, supporting "
        f"real-time pipelines and batch jobs at scale."
    )
    return impact_lines


def _regression_impact(r2, rmse, mae, target_col, domain):
    impact_lines = [
        f"**Forecasting Precision**: The model explains {r2 * 100:.1f}% of the variance in '{target_col}', "
        f"providing reliable forward-looking estimates with an average error of {mae:.4f} units.",
        f"**Planning Efficiency**: Accurate {target_col} predictions reduce over/under-provisioning, "
        f"directly cutting waste in {domain} operations.",
    ]
    if r2 >= 0.75:
        impact_lines.append(
            f"**Decision Confidence**: Stakeholders can use predicted values to set data-driven targets, "
            f"budgets, or thresholds — backed by a model that is statistically reliable (R²={r2:.3f})."
        )
    impact_lines.append(
        f"**Continuous Improvement**: As more data accumulates, retraining the model will further reduce RMSE ({rmse:.4f}) "
        f"and increase coverage of edge cases."
    )
    return impact_lines
