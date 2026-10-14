"""
src/survival.py
---------------
Survival analysis for StayWise using the lifelines library.

⚠️ LIMITATION: The IBM Telco dataset has no churn-date column.
   We use 'tenure' (months with the company) as the duration and
   'Churn' as the event indicator. This is a proxy, not exact timing.
   All outputs are clearly labelled as ESTIMATES.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def fit_kaplan_meier(df: pd.DataFrame) -> Any | None:
    """
    Fit a Kaplan-Meier survival model on the training dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'tenure' (int, months) and 'Churn' (0/1).

    Returns
    -------
    fitted KaplanMeierFitter or None if lifelines unavailable.
    """
    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=df["tenure"],
            event_observed=df["Churn"],
            label="Survival (Stay) Probability",
        )
        logger.info(
            "Kaplan-Meier fitted — median survival: %.1f months",
            kmf.median_survival_time_,
        )
        return kmf
    except ImportError:
        logger.warning("lifelines not installed — survival analysis unavailable.")
        return None
    except Exception as e:
        logger.error("KM fitting failed: %s", e)
        return None


def churn_probability_windows(
    kmf: Any,
    current_tenure: int,
    windows: list[int] | None = None,
) -> dict[str, float | str]:
    """
    Estimate churn probability within the next N days from current tenure.

    Since tenure is in months (30-day periods), we approximate:
        30 days ≈ 1 month, 60 days ≈ 2 months, 90 days ≈ 3 months.

    Parameters
    ----------
    kmf : fitted KaplanMeierFitter
    current_tenure : int — customer's current tenure in months
    windows : list[int] — day windows to estimate [30, 60, 90]

    Returns
    -------
    dict: {"30_day": float, "60_day": float, "90_day": float, "note": str}
    """
    if windows is None:
        windows = [30, 60, 90]

    result: dict[str, Any] = {
        "note": (
            "⚠️ Estimate only — based on Kaplan-Meier survival curve fitted to tenure. "
            "The dataset has no actual churn date column."
        )
    }

    if kmf is None:
        for w in windows:
            result[f"{w}_day"] = "N/A (lifelines not installed)"
        return result

    try:
        sf = kmf.survival_function_
        timeline = sf.index.values  # month values

        for w in windows:
            future_month = current_tenure + (w / 30)
            # P(churn by future_month | alive at current_tenure)
            # = 1 - S(future_month) / S(current_tenure)
            s_current = _survival_at(sf, current_tenure)
            s_future = _survival_at(sf, future_month)

            if s_current > 0:
                p_churn = max(0.0, 1.0 - (s_future / s_current))
            else:
                p_churn = 1.0

            result[f"{w}_day"] = round(float(p_churn), 4)

    except Exception as e:
        logger.error("Survival window computation failed: %s", e)
        for w in windows:
            result[f"{w}_day"] = "Error"

    return result


def _survival_at(sf: pd.DataFrame, t: float) -> float:
    """Interpolate survival function at time t."""
    col = sf.columns[0]
    timeline = sf.index.values
    values = sf[col].values
    if t <= timeline[0]:
        return float(values[0])
    if t >= timeline[-1]:
        return float(values[-1])
    return float(np.interp(t, timeline, values))


def plot_survival_curve(kmf: Any) -> go.Figure:
    """Plot the Kaplan-Meier survival curve."""
    fig = go.Figure()
    if kmf is None:
        fig.add_annotation(text="lifelines not installed", x=0.5, y=0.5, showarrow=False)
        return fig

    sf = kmf.survival_function_
    col = sf.columns[0]
    timeline = sf.index.values
    s_values = sf[col].values

    fig.add_trace(go.Scatter(
        x=timeline, y=s_values,
        mode="lines",
        name="Survival (Stay) Probability",
        line=dict(color="#0ea5e9", width=2),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.1)",
    ))

    # Confidence intervals
    try:
        ci = kmf.confidence_interval_survival_function_
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        fig.add_trace(go.Scatter(
            x=np.concatenate([timeline, timeline[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(14,165,233,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        ))
    except Exception:
        pass

    fig.update_layout(
        title="Kaplan-Meier Survival Curve (⚠️ Estimate — no churn-date in dataset)",
        xaxis_title="Tenure (months)",
        yaxis_title="Probability of Still Being a Customer",
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_risk_windows(windows: dict[str, Any], customer_tenure: int) -> go.Figure:
    """Bar chart showing 30/60/90-day churn probability windows."""
    labels = ["30-day", "60-day", "90-day"]
    keys = ["30_day", "60_day", "90_day"]
    values = []
    for k in keys:
        v = windows.get(k, 0)
        values.append(float(v) if isinstance(v, (int, float)) else 0.0)

    colors = ["#10b981" if v < 0.25 else "#f59e0b" if v < 0.60 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Estimated Churn Risk Windows (Current Tenure: {customer_tenure} months) — ⚠️ Estimate",
        xaxis_title="Time Window",
        yaxis_title="P(Churn) in Window",
        yaxis=dict(range=[0, 1]),
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
    )
    return fig
