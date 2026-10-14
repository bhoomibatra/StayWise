"""
tests/conftest.py — Shared pytest fixtures for StayWise tests.
All fixtures use synthetic data that matches the IBM Telco schema exactly.
No dataset file required for tests.
"""

import numpy as np
import pandas as pd
import pytest

# ─── Schema constants ─────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"

N_SAMPLES = 120  # enough for stratified 60/20/20 split


def _make_synthetic_df(n: int = N_SAMPLES, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic Telco-schema DataFrame with realistic value ranges."""
    rng = np.random.default_rng(seed)

    tenures = rng.integers(0, 73, size=n)
    monthly = rng.uniform(18, 120, size=n)

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], size=n),
        "SeniorCitizen": rng.integers(0, 2, size=n),
        "Partner": rng.choice(["Yes", "No"], size=n),
        "Dependents": rng.choice(["Yes", "No"], size=n),
        "tenure": tenures,
        "PhoneService": rng.choice(["Yes", "No"], size=n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
        "InternetService": rng.choice(["Fiber optic", "DSL", "No"], size=n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
        "PaymentMethod": rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], size=n),
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": (monthly * np.maximum(tenures, 1)).round(2),
        "Churn": rng.choice(["Yes", "No"], size=n, p=[0.27, 0.73]),
    })
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Raw synthetic DataFrame matching IBM Telco schema."""
    return _make_synthetic_df()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    """Cleaned DataFrame (after src.data.clean)."""
    from src.data import clean
    return clean(raw_df.copy())


@pytest.fixture(scope="session")
def engineered_df(clean_df):
    """Cleaned + feature-engineered DataFrame."""
    from src.features import engineer
    return engineer(clean_df.copy())


@pytest.fixture(scope="session")
def split_data(engineered_df):
    """Train/val/test split tuple."""
    from src.data import split
    return split(engineered_df)


@pytest.fixture(scope="session")
def trained_payload(split_data):
    """Full trained pipeline payload (pipeline + metadata)."""
    from src.train import train_on_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data
    return train_on_data(X_train, X_val, X_test, y_train, y_val, y_test)


@pytest.fixture
def single_record() -> dict:
    """A single customer record dict (no customerID, no Churn)."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.50,
        "TotalCharges": 1812.00,
    }
