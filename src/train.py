"""Model Training Module for StayWise ML Lab Project.

Implements model training, hyperparameter tuning via Stratified Cross-Validation,
and evaluation for Logistic Regression (Phase 3).
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    prepare_data,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Calculate standard binary classification evaluation metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (Churn = 1).

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and roc_auc.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def train_logistic_regression(
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    preprocessor=None,
) -> tuple[Pipeline, dict]:
    """Train and evaluate a Logistic Regression pipeline with 5-fold CV hyperparameter tuning.

    Steps:
    1. Bundle the unfitted preprocessor and Logistic Regression into a Scikit-learn Pipeline.
    2. Tune regularization strength C in [0.1, 1.0, 10.0] using 5-Fold Stratified CV on X_train.
    3. Fit the optimal pipeline on the full training set.
    4. Generate predictions and probability estimates on the holdout test set (X_test).
    5. Compute and report holdout metrics.

    Args:
        X_train: Optional pre-loaded training features.
        X_test: Optional pre-loaded test features.
        y_train: Optional pre-loaded training target.
        y_test: Optional pre-loaded test target.
        preprocessor: Optional unfitted ColumnTransformer.

    Returns:
        Tuple of (fitted_pipeline, metrics_dict).
    """
    # 1. Load and split data if not provided
    if any(arg is None for arg in (X_train, X_test, y_train, y_test, preprocessor)):
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # 2. Build unified Scikit-learn Pipeline (preprocessing + classifier)
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # 3. Define hyperparameter search grid for C (Inverse regularization strength)
    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
    }

    # 5-fold Stratified Cross-Validation on training data only
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=False,
    )

    print("=" * 60)
    print("STAYWISE — PHASE 3: LOGISTIC REGRESSION TRAINING")
    print("=" * 60)
    print("Running 5-fold Stratified Cross-Validation on training data...")
    grid_search.fit(X_train, y_train)

    print("\n--- Cross-Validation Results by C value ---")
    for mean_score, std_score, params in zip(
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"],
        grid_search.cv_results_["params"],
    ):
        c_val = params["classifier__C"]
        print(f"  C = {c_val:<4} -> Mean CV ROC-AUC: {mean_score:.4f} (+/- {std_score:.4f})")

    best_pipeline = grid_search.best_estimator_
    best_c = grid_search.best_params_["classifier__C"]
    print(f"\nOptimal Hyperparameter: C = {best_c}")

    # 4. Evaluate on holdout test set (X_test, y_test)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("\n--- Holdout Test Set Evaluation (1,409 rows) ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  (Overall correct classifications)")
    print(f"  Precision : {metrics['precision']:.4f}  (When predicting Churn, how often right)")
    print(f"  Recall    : {metrics['recall']:.4f}  (Out of actual churners, how many caught)")
    print(f"  F1-Score  : {metrics['f1']:.4f}  (Harmonic balance of precision & recall)")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}  (Ranking separation across all thresholds)")
    print("=" * 60)

    return best_pipeline, metrics


def train_knn(
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    preprocessor=None,
) -> tuple[Pipeline, dict]:
    """Train and evaluate a K-Nearest Neighbors (KNN) pipeline with 5-fold CV hyperparameter tuning.

    Steps:
    1. Bundle the unfitted preprocessor (with StandardScaler) and KNeighborsClassifier into a Pipeline.
    2. Tune neighbor count k in [5, 9, 15, 21] using 5-Fold Stratified CV on X_train.
    3. Fit the optimal KNN pipeline on the full training set.
    4. Predict classes and class probabilities on the holdout test set (X_test).
    5. Compute and report holdout metrics.

    Args:
        X_train: Optional pre-loaded training features.
        X_test: Optional pre-loaded test features.
        y_train: Optional pre-loaded training target.
        y_test: Optional pre-loaded test target.
        preprocessor: Optional unfitted ColumnTransformer.

    Returns:
        Tuple of (fitted_pipeline, metrics_dict).
    """
    if any(arg is None for arg in (X_train, X_test, y_train, y_test, preprocessor)):
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    # Search space: odd k values to prevent tie votes in binary classification
    param_grid = {
        "classifier__n_neighbors": [5, 9, 15, 21],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=False,
    )

    print("=" * 60)
    print("STAYWISE — PHASE 4: K-NEAREST NEIGHBORS (KNN) TRAINING")
    print("=" * 60)
    print("Running 5-fold Stratified Cross-Validation on training data...")
    grid_search.fit(X_train, y_train)

    print("\n--- Cross-Validation Results by k value ---")
    for mean_score, std_score, params in zip(
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"],
        grid_search.cv_results_["params"],
    ):
        k_val = params["classifier__n_neighbors"]
        print(f"  k = {k_val:<3} -> Mean CV ROC-AUC: {mean_score:.4f} (+/- {std_score:.4f})")

    best_pipeline = grid_search.best_estimator_
    best_k = grid_search.best_params_["classifier__n_neighbors"]
    print(f"\nOptimal Hyperparameter: k = {best_k}")

    # Evaluate on holdout test set (X_test, y_test)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("\n--- Holdout Test Set Evaluation (1,409 rows) ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  (Overall correct classifications)")
    print(f"  Precision : {metrics['precision']:.4f}  (When predicting Churn, how often right)")
    print(f"  Recall    : {metrics['recall']:.4f}  (Out of actual churners, how many caught)")
    print(f"  F1-Score  : {metrics['f1']:.4f}  (Harmonic balance of precision & recall)")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}  (Ranking separation across all thresholds)")
    print("=" * 60)

    return best_pipeline, metrics


def compare_knn_scaling(
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
) -> dict:
    """Demonstrate why feature scaling (StandardScaler) is mandatory for KNN.

    Compares test metrics of KNN trained with StandardScaler vs without StandardScaler.
    """
    if any(arg is None for arg in (X_train, X_test, y_train, y_test)):
        X_train, X_test, y_train, y_test, _ = prepare_data()

    # Preprocessor WITHOUT StandardScaler for numerical columns
    unscaled_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERICAL_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    unscaled_pipeline = Pipeline(
        [
            ("preprocessor", unscaled_preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=21)),
        ]
    )
    unscaled_pipeline.fit(X_train, y_train)
    unscaled_pred = unscaled_pipeline.predict(X_test)
    unscaled_prob = unscaled_pipeline.predict_proba(X_test)[:, 1]
    unscaled_metrics = compute_metrics(y_test, unscaled_pred, unscaled_prob)

    print("\n--- Impact of Feature Scaling on KNN (k=21) ---")
    print(f"  WITHOUT Scaling ROC-AUC : {unscaled_metrics['roc_auc']:.4f} (Recall: {unscaled_metrics['recall']:.4f})")

    return unscaled_metrics


if __name__ == "__main__":
    train_logistic_regression()
    train_knn()
    compare_knn_scaling()

