"""
src/evaluate.py
---------------
Compute classification metrics and generate evaluation plots.

All plotting functions return Plotly figures for use in the Streamlit app.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_pred : array-like of 0/1 (thresholded predictions)
    y_prob : array-like of floats in [0, 1] (positive class probabilities)
    prefix : str
        Prepended to each key (e.g. 'val_', 'test_').

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    return {
        f"{prefix}accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        f"{prefix}precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        f"{prefix}recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f"{prefix}f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        f"{prefix}roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        f"{prefix}pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
    }


# ─── Plotly figures ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
) -> go.Figure:
    """Return a Plotly heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Churn (0)", "Churn (1)"]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual"},
        title=title,
    )
    fig.update_layout(
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_roc_curves(curves: dict[str, dict]) -> go.Figure:
    """
    Plot overlaid ROC curves for multiple models.

    Parameters
    ----------
    curves : dict[model_name, {"fpr": array, "tpr": array, "auc": float}]
    """
    fig = go.Figure()
    colors = ["#0ea5e9", "#f59e0b", "#10b981", "#ef4444"]
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines",
            name=f"{name} (AUC={data['auc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Random", showlegend=True,
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_pr_curves(curves: dict[str, dict]) -> go.Figure:
    """
    Plot overlaid Precision-Recall curves for multiple models.

    Parameters
    ----------
    curves : dict[model_name, {"precision": array, "recall": array, "auc": float}]
    """
    fig = go.Figure()
    colors = ["#0ea5e9", "#f59e0b", "#10b981", "#ef4444"]
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["recall"], y=data["precision"],
            mode="lines",
            name=f"{name} (AUC={data['auc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Return a reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode="lines+markers",
        name=model_name, line=dict(color="#0ea5e9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Perfect calibration",
    ))
    fig.update_layout(
        title=f"Calibration Plot — {model_name}",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_profit_vs_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    offer_cost: float = 50.0,
    success_rate: float = 0.30,
    current_threshold: float = 0.5,
) -> go.Figure:
    """Plot expected profit as a function of classification threshold."""
    thresholds = np.linspace(0.01, 0.99, 99)
    profits = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        profits.append(tp * success_rate - (tp + fp) * (offer_cost / 100))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=profits, mode="lines",
        name="Expected Profit", line=dict(color="#10b981", width=2),
    ))
    fig.add_vline(
        x=current_threshold, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Current: {current_threshold:.2f}",
    )
    fig.update_layout(
        title="Profit vs Classification Threshold (Validation Set)",
        xaxis_title="Threshold",
        yaxis_title="Expected Profit (relative units)",
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
    )
    return fig


def compute_curves_for_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Return ROC and PR curve data for a single model."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "auc": round(float(average_precision_score(y_true, y_prob)), 4),
        },
    }


def plot_feature_importance_bar(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of feature importances (top N)."""
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    fig = px.bar(
        df.sort_values("importance"),
        x="importance", y="feature",
        orientation="h",
        title=title,
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        yaxis_title="",
        xaxis_title="Importance Score",
    )
    return fig
