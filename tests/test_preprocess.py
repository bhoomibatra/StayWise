"""Unit tests for StayWise data preprocessing and loading pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data
from src.preprocess import prepare_data, create_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def test_raw_data_loading():
    """Verify raw data loading, row count, and TotalCharges conversion."""
    df = load_raw_data(DEFAULT_DATA_PATH)
    assert len(df) == 7043, f"Expected 7,043 rows, got {len(df)}"
    assert "TotalCharges" in df.columns
    # Check TotalCharges is numeric float
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    # Exactly 11 whitespace values converted to NaN
    assert df["TotalCharges"].isna().sum() == 11


def test_stratified_train_test_split():
    """Verify train/test split proportions and stratified churn distribution."""
    X_train, X_test, y_train, y_test, _ = prepare_data(test_size=0.20, random_state=42)

    assert len(X_train) == 5634, f"Expected 5,634 train rows, got {len(X_train)}"
    assert len(X_test) == 1409, f"Expected 1,409 test rows, got {len(X_test)}"

    # Ensure customerID is excluded
    assert "customerID" not in X_train.columns
    assert "customerID" not in X_test.columns
    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns

    # Verify stratified balance (both close to 26.54%)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - 0.2654) < 0.01, f"Train churn rate {train_rate} deviates from expected"
    assert abs(test_rate - 0.2654) < 0.01, f"Test churn rate {test_rate} deviates from expected"
    assert abs(train_rate - test_rate) < 0.005, "Train and test churn rates must be closely matched"


def test_preprocessor_transformation_no_nans():
    """Verify ColumnTransformer outputs complete numeric arrays without NaNs."""
    X_train, X_test, _, _, preprocessor = prepare_data()

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    assert not np.isnan(X_train_trans).any(), "Transformed X_train contains NaN values"
    assert not np.isnan(X_test_trans).any(), "Transformed X_test contains NaN values"
    assert X_train_trans.shape[0] == 5634
    assert X_test_trans.shape[0] == 1409
    assert X_train_trans.shape[1] == X_test_trans.shape[1]
