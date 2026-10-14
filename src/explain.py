"""
src/explain.py
--------------
SHAP-based global and per-customer explainability for StayWise.

⚠️ SHAP values describe *model behaviour*, not causal proof.
   Feature importance shows what the model used — not what causes churn.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ─── Plain-English feature labels ─────────────────────────────────────────────
# Maps engineered/raw column names → human-readable reason strings
FEATURE_LABELS: dict[str, str] = {
    "Contract":              "Contract type",
    "is_month_to_month":     "Month-to-month contract",
    "tenure":                "Length of time as customer",
    "tenure_group":          "Customer tenure group",
    "MonthlyCharges":        "Monthly bill amount",
    "TotalCharges":          "Total spend to date",
    "charges_per_month":     "Average effective monthly charge",
    "InternetService":       "Internet service type",
    "TechSupport":           "No tech support",
    "OnlineSecurity":        "No online security",
    "OnlineBackup":          "No online backup",
    "DeviceProtection":      "No device protection",
    "StreamingTV":           "Streaming TV subscription",
    "StreamingMovies":       "Streaming movies subscription",
    "PaymentMethod":         "Payment method",
    "PaperlessBilling":      "Paperless billing",
    "num_addons":            "Number of add-on services",
    "SeniorCitizen":         "Senior citizen",
    "Partner":               "Has a partner",
    "Dependents":            "Has dependents",
    "gender":                "Gender",
    "PhoneService":          "Phone service",
    "MultipleLines":         "Multiple phone lines",
}

# Positive SHAP = increases churn risk; these templates reflect that direction
REASON_TEMPLATES: dict[str, str] = {
    "is_month_to_month":  "Month-to-month contract (easy to cancel)",
    "MonthlyCharges":     "High monthly bill",
    "tenure":             "Short tenure (new customer)",
    "charges_per_month":  "High effective monthly charge",
    "TechSupport":        "No tech support subscription",
    "OnlineSecurity":     "No online security subscription",
    "InternetService":    "Fiber optic service (higher churn segment)",
    "num_addons":         "Few add-on services (low engagement)",
    "Contract":           "Short-term or no contract commitment",
    "PaymentMethod":      "Electronic check payment method",
    "PaperlessBilling":   "Paperless billing (associated with higher churn)",
    "SeniorCitizen":      "Senior citizen segment",
    "tenure_group":       "Early tenure stage",
}


def _get_shap_explainer(pipeline: Any, X_background: pd.DataFrame) -> Any:
    """
    Auto-select the appropriate SHAP explainer for the fitted pipeline.

    Tries TreeExplainer first, falls back to LinearExplainer, then KernelExplainer.
    """
    import shap

    try:
        # For calibrated pipelines, access the underlying estimator
        estimator = _unwrap_pipeline(pipeline)

        # Check if it's a tree-based model
        tree_types = ("XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier",
                      "GradientBoostingClassifier")
        if type(estimator).__name__ in tree_types:
            preprocessor = _get_preprocessor(pipeline)
            X_transformed = preprocessor.transform(X_background)
            return shap.TreeExplainer(estimator), X_transformed, True

        # Linear model
        linear_types = ("LogisticRegression", "LinearSVC", "Ridge", "Lasso")
        if type(estimator).__name__ in linear_types:
            preprocessor = _get_preprocessor(pipeline)
            X_transformed = preprocessor.transform(X_background)
            return shap.LinearExplainer(estimator, X_transformed), X_transformed, True

    except Exception as e:
        logger.warning("Primary SHAP explainer failed (%s). Using KernelExplainer.", e)

    # Fallback: KernelExplainer (slower but universal)
    try:
        import shap
        preprocessor = _get_preprocessor(pipeline)
        X_transformed = preprocessor.transform(X_background.head(50))
        predict_fn = lambda x: pipeline.predict_proba(  # noqa: E731
            pd.DataFrame(x, columns=X_background.columns[:x.shape[1]])
        )[:, 1]
        return shap.KernelExplainer(predict_fn, X_transformed), X_transformed, False
    except Exception as e:
        logger.error("All SHAP explainers failed: %s", e)
        return None, None, False


def _unwrap_pipeline(pipeline: Any) -> Any:
    """Extract the base classifier from a (possibly calibrated) sklearn pipeline."""
    # CalibratedClassifierCV wraps an estimator in .estimator
    if hasattr(pipeline, "calibrated_classifiers_"):
        inner = pipeline.calibrated_classifiers_[0].estimator
    elif hasattr(pipeline, "estimator"):
        inner = pipeline.estimator
    else:
        inner = pipeline

    # sklearn Pipeline: last step
    if hasattr(inner, "named_steps"):
        return inner.named_steps[list(inner.named_steps.keys())[-1]]
    return inner


def _get_preprocessor(pipeline: Any) -> Any:
    """Extract the preprocessing ColumnTransformer from a (possibly calibrated) pipeline."""
    if hasattr(pipeline, "calibrated_classifiers_"):
        inner = pipeline.calibrated_classifiers_[0].estimator
    elif hasattr(pipeline, "estimator"):
        inner = pipeline.estimator
    else:
        inner = pipeline

    if hasattr(inner, "named_steps"):
        return inner.named_steps["preprocessor"]
    raise ValueError("Cannot find preprocessor in pipeline.")


def get_feature_names_from_preprocessor(preprocessor: Any) -> list[str]:
    """Extract feature names after one-hot encoding from the ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(100)]


def compute_global_importance(
    pipeline: Any,
    X_sample: pd.DataFrame,
    max_samples: int = 200,
) -> pd.DataFrame | None:
    """
    Compute mean |SHAP| global feature importance.

    Parameters
    ----------
    pipeline : fitted pipeline (possibly CalibratedClassifierCV)
    X_sample : pd.DataFrame — background dataset (uses up to max_samples rows)
    max_samples : int

    Returns
    -------
    pd.DataFrame with columns ['feature', 'importance'] sorted descending,
    or None if SHAP computation fails.
    """
    warnings.filterwarnings("ignore")
    X_bg = X_sample.sample(min(max_samples, len(X_sample)), random_state=42)

    explainer, X_transformed, is_matrix = _get_shap_explainer(pipeline, X_bg)
    if explainer is None:
        return None

    try:
        shap_values = explainer.shap_values(X_transformed)
        # Binary classification: shap_values is list [class0, class1] or just class1
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1])
        else:
            sv = np.array(shap_values)

        mean_abs = np.abs(sv).mean(axis=0)

        try:
            preprocessor = _get_preprocessor(pipeline)
            feature_names = get_feature_names_from_preprocessor(preprocessor)
        except Exception:
            feature_names = [f"f{i}" for i in range(len(mean_abs))]

        n = min(len(feature_names), len(mean_abs))
        df = pd.DataFrame({
            "feature": feature_names[:n],
            "importance": mean_abs[:n],
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    except Exception as e:
        logger.error("Global SHAP computation failed: %s", e)
        return None


def explain_customer(
    pipeline: Any,
    record: pd.DataFrame,
    top_n: int = 4,
) -> list[dict[str, Any]]:
    """
    Compute per-customer SHAP values and map to plain-English reasons.

    Parameters
    ----------
    pipeline : fitted pipeline
    record : pd.DataFrame — single-row feature dataframe (after engineering)
    top_n : int — number of top drivers to return

    Returns
    -------
    list of dicts: [{"feature": str, "shap_value": float, "reason": str, "direction": str}]
    or [] if SHAP fails (caller should handle gracefully).
    """
    warnings.filterwarnings("ignore")

    explainer, X_transformed, _ = _get_shap_explainer(pipeline, record)
    if explainer is None:
        return []

    try:
        row_transformed = _get_preprocessor(pipeline).transform(record)
        shap_values = explainer.shap_values(row_transformed)

        if isinstance(shap_values, list):
            sv = np.array(shap_values[1])[0]
        else:
            sv = np.array(shap_values)[0]

        try:
            preprocessor = _get_preprocessor(pipeline)
            feature_names = get_feature_names_from_preprocessor(preprocessor)
        except Exception:
            feature_names = [f"f{i}" for i in range(len(sv))]

        n = min(len(feature_names), len(sv))
        pairs = sorted(zip(feature_names[:n], sv[:n]), key=lambda x: abs(x[1]), reverse=True)

        results = []
        for feat, val in pairs[:top_n]:
            base_feat = feat.split("__")[-1] if "__" in feat else feat
            reason = _get_reason(base_feat, val)
            results.append({
                "feature": feat,
                "base_feature": base_feat,
                "shap_value": round(float(val), 4),
                "reason": reason,
                "direction": "increases risk" if val > 0 else "reduces risk",
            })
        return results

    except Exception as e:
        logger.error("Per-customer SHAP explanation failed: %s", e)
        return []


def _get_reason(feature: str, shap_val: float) -> str:
    """Map a feature name and SHAP direction to a plain-English reason."""
    # Try exact match first, then prefix match
    if feature in REASON_TEMPLATES and shap_val > 0:
        return REASON_TEMPLATES[feature]
    label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
    direction = "↑ increases" if shap_val > 0 else "↓ reduces"
    return f"{label} {direction} churn risk"


def plot_global_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of global SHAP importance."""
    df = importance_df.head(top_n).sort_values("importance")
    fig = px.bar(
        df, x="importance", y="feature",
        orientation="h",
        title=f"Global Feature Importance (Top {top_n}) — Mean |SHAP|",
        color="importance",
        color_continuous_scale="Blues",
        labels={"importance": "Mean |SHAP Value|", "feature": ""},
    )
    fig.update_layout(
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
    )
    return fig


def plot_waterfall(drivers: list[dict]) -> go.Figure:
    """Waterfall chart for a single customer's SHAP drivers."""
    if not drivers:
        return go.Figure()

    features = [d["reason"] for d in drivers]
    values = [d["shap_value"] for d in drivers]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Key Factors for This Customer (SHAP Contribution)",
        xaxis_title="SHAP Value (positive = higher churn risk)",
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        height=300 + len(drivers) * 30,
    )
    return fig
