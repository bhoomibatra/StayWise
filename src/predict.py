"""
src/predict.py
--------------
Single-record and batch prediction API using the saved StayWise pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features import engineer


def _load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_pipeline(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load the saved pipeline and metadata from disk.

    Returns
    -------
    dict with keys: 'pipeline' (fitted estimator), 'metadata' (dict)

    Raises
    ------
    FileNotFoundError if models/best_pipeline.joblib does not exist.
    """
    cfg = _load_config(config_path)
    pipeline_path = _project_root() / cfg["paths"]["best_pipeline"]

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"\n  ❌ No trained model found at: {pipeline_path}\n"
            "  Run  python -m src.train  to train the model first.\n"
        )
    return joblib.load(pipeline_path)


def predict_single(
    record: dict[str, Any],
    payload: dict[str, Any] | None = None,
    threshold: float | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    record : dict
        Feature values keyed by column name. customerID and Churn are ignored.
    payload : dict | None
        Pre-loaded pipeline payload (avoids repeated disk I/O in the app).
    threshold : float | None
        Classification cutoff. Uses config default if None.
    config_path : Path | None

    Returns
    -------
    dict with keys:
        probability (float), label (int), risk_category (str), threshold (float)
    """
    if payload is None:
        payload = load_pipeline(config_path)

    cfg = _load_config(config_path)
    if threshold is None:
        threshold = payload["metadata"].get("threshold", cfg["business"]["threshold_default"])

    # Build single-row DataFrame and apply feature engineering
    df = pd.DataFrame([record])
    df = _drop_non_features(df)
    df = _apply_engineering(df)

    pipeline = payload["pipeline"]
    prob = float(pipeline.predict_proba(df)[:, 1][0])
    label = int(prob >= threshold)

    return {
        "probability": round(prob, 4),
        "label": label,
        "risk_category": _risk_category(prob, cfg),
        "threshold": threshold,
    }


def predict_batch(
    df: pd.DataFrame,
    payload: dict[str, Any] | None = None,
    threshold: float | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """
    Predict churn probability for a batch of customers.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe. customerID is preserved if present. Churn is ignored.
    payload : dict | None
        Pre-loaded pipeline payload.
    threshold : float | None
    config_path : Path | None

    Returns
    -------
    pd.DataFrame with columns:
        customerID (if present), churn_probability, churn_label, risk_category
    """
    if payload is None:
        payload = load_pipeline(config_path)

    cfg = _load_config(config_path)
    if threshold is None:
        threshold = payload["metadata"].get("threshold", cfg["business"]["threshold_default"])

    customer_ids = df["customerID"].copy() if "customerID" in df.columns else None

    features = _drop_non_features(df.copy())
    features = _apply_engineering(features)

    pipeline = payload["pipeline"]
    probs = pipeline.predict_proba(features)[:, 1]
    labels = (probs >= threshold).astype(int)
    categories = [_risk_category(p, cfg) for p in probs]

    result = pd.DataFrame({
        "churn_probability": np.round(probs, 4),
        "churn_label": labels,
        "risk_category": categories,
    })

    if customer_ids is not None:
        result.insert(0, "customerID", customer_ids.values)

    return result


def validate_batch_schema(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """
    Validate that all required feature columns are present.

    Returns
    -------
    list of missing column names (empty if all present).
    """
    return [c for c in required_cols if c not in df.columns]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop customerID and Churn columns if present."""
    return df.drop(columns=["customerID", "Churn"], errors="ignore")


def _apply_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering, handling missing columns gracefully."""
    try:
        return engineer(df)
    except Exception:
        # Return as-is; the pipeline's OHE handles unknown categories
        return df


def _risk_category(prob: float, cfg: dict) -> str:
    """Map a churn probability to a labelled risk tier."""
    cats = cfg["risk_categories"]
    if prob < cats["low"][1]:
        return "Low"
    elif prob < cats["medium"][1]:
        return "Medium"
    elif prob < cats["high"][1]:
        return "High"
    else:
        return "Critical"
