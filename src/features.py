"""
src/features.py
---------------
Feature engineering for the IBM Telco Customer Churn dataset.

All features are derived from existing columns only.
No external data or fabricated values are introduced.
"""

from __future__ import annotations

import pandas as pd

# Add-on service columns used to compute num_addons
ADDON_COLS: list[str] = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Numeric features (before engineering)
NUMERIC_FEATURES: list[str] = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "charges_per_month", "num_addons",
]

# Categorical features (before engineering; tenure_group added after)
CATEGORICAL_FEATURES: list[str] = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the Telco dataframe.

    New Columns
    -----------
    tenure_group : str
        Ordinal tenure bucket: '0-12', '13-24', '25-48', '49-60', '61+'
    charges_per_month : float
        TotalCharges / max(tenure, 1)  — effective average monthly payment.
        Helps distinguish new high-spenders from long-term stable accounts.
    num_addons : int
        Count of add-on services subscribed (0–6).
        Proxy for product engagement / switching cost.
    is_month_to_month : int
        1 if Contract == 'Month-to-month', else 0.
        Strongest single predictor of churn in this dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (customerID already dropped, Churn encoded).

    Returns
    -------
    pd.DataFrame
        Dataframe with 4 new columns appended. Original columns are unchanged.
    """
    df = df.copy()

    df["tenure_group"] = _tenure_group(df["tenure"])
    df["charges_per_month"] = _charges_per_month(df)
    df["num_addons"] = _num_addons(df)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    return df


# ─── Private helpers ──────────────────────────────────────────────────────────

def _tenure_group(tenure: pd.Series) -> pd.Series:
    """Bin tenure (months) into labelled groups."""
    bins = [0, 12, 24, 48, 60, float("inf")]
    labels = ["0-12", "13-24", "25-48", "49-60", "61+"]
    return pd.cut(tenure, bins=bins, labels=labels, right=True, include_lowest=True).astype(str)


def _charges_per_month(df: pd.DataFrame) -> pd.Series:
    """
    TotalCharges / max(tenure, 1).

    Handles TotalCharges NaN (new customers with tenure=0) gracefully:
    result will be NaN and will be imputed downstream.
    """
    denom = df["tenure"].clip(lower=1)
    return (df["TotalCharges"] / denom).round(4)


def _num_addons(df: pd.DataFrame) -> pd.Series:
    """Count how many of the 6 add-on services are subscribed ('Yes')."""
    present = [c for c in ADDON_COLS if c in df.columns]
    return (df[present] == "Yes").sum(axis=1).astype(int)


def get_feature_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return numeric and categorical feature lists based on the dataframe's columns.

    Used by the preprocessing pipeline builder in train.py.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe (Churn column excluded).

    Returns
    -------
    dict with keys 'numeric' and 'categorical'.
    """
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    # Add is_month_to_month as numeric (binary 0/1)
    if "is_month_to_month" in df.columns:
        numeric.append("is_month_to_month")
    return {"numeric": numeric, "categorical": categorical}
