"""
src/data.py
-----------
Load, validate, clean, and split the IBM Telco Customer Churn dataset.

All preprocessing is documented; no target leakage is introduced here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ─── Column definitions ────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


# ─── Config helper ────────────────────────────────────────────────────────────

def _load_config(config_path: Path | None = None) -> dict:
    """Load config.yaml from project root."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ─── Public API ───────────────────────────────────────────────────────────────

def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the raw Telco CSV file.

    Parameters
    ----------
    path : str | Path | None
        Path to CSV. Defaults to the path in config.yaml.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataframe.

    Raises
    ------
    FileNotFoundError
        With clear instructions if the file is missing.
    """
    if path is None:
        cfg = _load_config()
        path = _project_root() / cfg["paths"]["raw_data"]

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\n\n  ❌ Dataset not found at: {path}\n\n"
            "  To fix this:\n"
            "  1. Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "  2. Place the file at:\n"
            f"     {path}\n"
        )

    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows × %d columns", len(df), df.shape[1])
    return df


def validate_schema(df: pd.DataFrame, required_cols: list[str] | None = None) -> None:
    """
    Validate that all required columns are present.

    Raises
    ------
    ValueError
        Names every missing column.
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} required column(s): {missing}\n"
            "Check that you are using the IBM Telco Customer Churn CSV."
        )
    logger.info("Schema validation passed (%d columns present)", len(required_cols))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco dataframe.

    Steps
    -----
    1. Validate schema.
    2. Convert TotalCharges (string with blanks) to float; blank → NaN.
    3. Drop customerID.
    4. Encode Churn Yes/No → 1/0.
    5. Reset index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering and splitting.
    """
    df = df.copy()
    validate_schema(df)

    # 1. TotalCharges: coerce blanks / non-numeric to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = df["TotalCharges"].isna().sum()
    if n_blank > 0:
        logger.info("TotalCharges: %d blank/non-numeric values → NaN (will be imputed)", n_blank)

    # 2. Drop identifier
    df = df.drop(columns=["customerID"], errors="ignore")

    # 3. Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isna().any():
        raise ValueError("Churn column contains unexpected values after encoding.")

    # 4. Report data quality
    _report_quality(df)

    df = df.reset_index(drop=True)
    return df


def _report_quality(df: pd.DataFrame) -> None:
    """Log data quality summary."""
    n_dup = df.duplicated().sum()
    n_missing = df.isna().sum().sum()
    churn_rate = df["Churn"].mean()
    logger.info(
        "Data quality — rows: %d | duplicates: %d | missing values: %d | churn rate: %.1f%%",
        len(df), n_dup, n_missing, churn_rate * 100,
    )
    if n_dup > 0:
        logger.warning("Found %d duplicate rows — consider reviewing.", n_dup)


def get_quality_report(df: pd.DataFrame) -> dict:
    """
    Return a dictionary with data quality statistics for display in the app.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    dict
        Keys: n_rows, n_cols, n_duplicates, n_missing_total, churn_rate,
              missing_by_col (dict), dtypes_summary (dict).
    """
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "n_missing_total": int(df.isna().sum().sum()),
        "churn_rate": float(df["Churn"].mean()),
        "missing_by_col": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "class_distribution": df["Churn"].value_counts().to_dict(),
    }


def split(
    df: pd.DataFrame,
    config_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Stratified train / validation / test split.

    Splits follow config.yaml: test_size=0.20, val_size=0.20 of remainder.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned (and optionally feature-engineered) dataframe.
    config_path : Path | None
        Optional override for config path.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    cfg = _load_config(config_path)
    seed = cfg["data"]["random_state"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Step 1: hold out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Step 2: split remaining into train and validation
    # val_size is fraction of original dataset, so adjust
    val_fraction_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction_of_temp,
        random_state=seed,
        stratify=y_temp,
    )

    logger.info(
        "Split — train: %d | val: %d | test: %d | churn rates: %.1f%% / %.1f%% / %.1f%%",
        len(X_train), len(X_val), len(X_test),
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
