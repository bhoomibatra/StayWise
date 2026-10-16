"""Unit tests for StayWise model pipelines, predictions, and model reloading."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.preprocess import prepare_data
from src.train import (
    BEST_MODEL_PATH,
    load_model,
    train_knn,
    train_logistic_regression,
    train_random_forest,
    train_svm,
)


@pytest.fixture(scope="module")
def dataset_splits():
    """Load train and test splits once for all model tests."""
    return prepare_data()


def test_logistic_regression_pipeline(dataset_splits):
    """Test Logistic Regression pipeline output validity."""
    X_train, X_test, y_train, y_test, preprocessor = dataset_splits
    pipeline, metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test, preprocessor
    )
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert metrics["roc_auc"] > 0.80
    assert metrics["accuracy"] > 0.75


def test_knn_pipeline(dataset_splits):
    """Test KNN pipeline output validity."""
    X_train, X_test, y_train, y_test, preprocessor = dataset_splits
    pipeline, metrics = train_knn(
        X_train, X_test, y_train, y_test, preprocessor
    )
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert metrics["roc_auc"] > 0.78


def test_random_forest_pipeline(dataset_splits):
    """Test Random Forest pipeline output validity."""
    X_train, X_test, y_train, y_test, preprocessor = dataset_splits
    pipeline, metrics, top_features = train_random_forest(
        X_train, X_test, y_train, y_test, preprocessor
    )
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert len(top_features) == 10
    assert metrics["roc_auc"] > 0.80


def test_svm_pipeline(dataset_splits):
    """Test Support Vector Machine (SVM) pipeline output validity."""
    X_train, X_test, y_train, y_test, preprocessor = dataset_splits
    pipeline, metrics = train_svm(
        X_train, X_test, y_train, y_test, preprocessor
    )
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert metrics["roc_auc"] > 0.78
    assert metrics["accuracy"] > 0.75
    assert metrics["f1"] > 0.50


def test_best_model_reload_and_prediction():
    """Verify saved best model loads from disk and predicts on raw customer input."""
    assert BEST_MODEL_PATH.exists(), f"Saved model not found at {BEST_MODEL_PATH}"
    pipeline = load_model(BEST_MODEL_PATH)

    # Raw customer record
    sample_customer = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 700.0,
            }
        ]
    )

    pred = pipeline.predict(sample_customer)
    prob = pipeline.predict_proba(sample_customer)

    assert pred[0] in (0, 1)
    assert 0.0 <= prob[0][1] <= 1.0
    assert abs(prob[0].sum() - 1.0) < 1e-5


def test_edge_case_inputs():
    """Test edge case customer inputs (tenure=0, zero charges, maximum charges)."""
    assert BEST_MODEL_PATH.exists()
    pipeline = load_model(BEST_MODEL_PATH)

    edge_cases = pd.DataFrame(
        [
            {
                # New subscriber with tenure=0 and TotalCharges missing/0
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 20.0,
                "TotalCharges": 0.0,
            },
            {
                # Extreme tenure subscriber with high charges
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 72,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 118.75,
                "TotalCharges": 8684.80,
            },
        ]
    )

    preds = pipeline.predict(edge_cases)
    probs = pipeline.predict_proba(edge_cases)

    assert len(preds) == 2
    assert len(probs) == 2
    for p in probs:
        assert 0.0 <= p[1] <= 1.0


def test_dataset_scoring_and_risk_categorization():
    """Verify that full-dataset batch scoring generates valid predictions, probabilities, and risk tiers."""
    from app import extract_risk_drivers, get_retention_playbook, run_dataset_scoring

    scored_df, err = run_dataset_scoring()
    assert err is None, f"Scoring error: {err}"
    assert scored_df is not None
    assert len(scored_df) == 7043

    # Predictions & probabilities validation
    assert set(scored_df["predicted_churn"].unique()).issubset({"Churn", "Stay"})
    assert set(scored_df["risk_category"].unique()).issubset({"High Risk", "Medium Risk", "Low Risk"})
    assert scored_df["churn_probability"].isna().sum() == 0
    assert ((scored_df["churn_probability"] >= 0.0) & (scored_df["churn_probability"] <= 1.0)).all()

    # Verify ID column preservation
    assert "customerID" in scored_df.columns
    assert "Churn" in scored_df.columns

    # Verify risk driver extraction and retention playbook
    first_customer = scored_df.iloc[0]
    drivers = extract_risk_drivers(first_customer)
    assert isinstance(drivers, list)
    assert len(drivers) > 0

    playbook = get_retention_playbook(first_customer, "High Risk")
    assert isinstance(playbook, list)
    assert len(playbook) > 0

