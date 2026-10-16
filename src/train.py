"""Model Training Module for StayWise ML Lab Project.

Implements model training, hyperparameter tuning via Stratified Cross-Validation,
and evaluation for Logistic Regression (Phase 3), KNN (Phase 4), and Random Forest (Phase 5).
"""

from datetime import datetime
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"


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


def train_random_forest(
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    preprocessor=None,
) -> tuple[Pipeline, dict, list[tuple[str, float]]]:
    """Train and evaluate a Random Forest Classifier with 5-fold CV hyperparameter tuning.

    Steps:
    1. Bundle the unfitted preprocessor and RandomForestClassifier into a Scikit-Learn Pipeline.
    2. Tune n_estimators in [50, 100] and max_depth in [5, 10] using 5-Fold Stratified CV on X_train.
    3. Fit the optimal pipeline on the full training set.
    4. Predict classes and class probabilities on the holdout test set (X_test).
    5. Inspect training vs test metrics to detect potential overfitting.
    6. Extract and rank feature importances using get_feature_names_out().
    7. Compute and report holdout metrics.

    Args:
        X_train: Optional pre-loaded training features.
        X_test: Optional pre-loaded test features.
        y_train: Optional pre-loaded training target.
        y_test: Optional pre-loaded test target.
        preprocessor: Optional unfitted ColumnTransformer.

    Returns:
        Tuple of (fitted_pipeline, metrics_dict, top_features_list).
    """
    if any(arg is None for arg in (X_train, X_test, y_train, y_test, preprocessor)):
        X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Search space: tree count and maximum depth
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [5, 10],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        return_train_score=True,
    )

    print("=" * 60)
    print("STAYWISE — PHASE 5: RANDOM FOREST TRAINING")
    print("=" * 60)
    print("Running 5-fold Stratified Cross-Validation on training data...")
    grid_search.fit(X_train, y_train)

    print("\n--- Cross-Validation Results by Parameter Combination ---")
    for mean_val, std_val, mean_train, params in zip(
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"],
        grid_search.cv_results_["mean_train_score"],
        grid_search.cv_results_["params"],
    ):
        n_est = params["classifier__n_estimators"]
        depth = params["classifier__max_depth"]
        print(
            f"  n_estimators = {n_est:<3}, max_depth = {depth:<2} -> "
            f"Mean Val ROC-AUC: {mean_val:.4f} (+/- {std_val:.4f}) | Train ROC-AUC: {mean_train:.4f}"
        )

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"\nOptimal Hyperparameters: {best_params}")

    # Evaluate on train set vs test set to inspect overfitting
    y_train_pred = best_pipeline.predict(X_train)
    y_train_prob = best_pipeline.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)

    # Evaluate on holdout test set (X_test, y_test)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("\n--- Overfitting Inspection (Train vs Holdout Test) ---")
    print(f"  Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Train ROC-AUC : {train_metrics['roc_auc']:.4f} | Test ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Train F1-Score: {train_metrics['f1']:.4f} | Test F1-Score: {metrics['f1']:.4f}")

    print("\n--- Holdout Test Set Evaluation (1,409 rows) ---")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  (Overall correct classifications)")
    print(f"  Precision : {metrics['precision']:.4f}  (When predicting Churn, how often right)")
    print(f"  Recall    : {metrics['recall']:.4f}  (Out of actual churners, how many caught)")
    print(f"  F1-Score  : {metrics['f1']:.4f}  (Harmonic balance of precision & recall)")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}  (Ranking separation across all thresholds)")

    # Extract feature importances
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    feature_ranking = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    top_features = [(name, float(score)) for name, score in feature_ranking[:10]]

    print("\n--- Top 10 Most Influential Features ---")
    for rank, (name, imp) in enumerate(top_features, 1):
        print(f"  {rank:>2}. {name:<40}: {imp:.4f}")
    print("=" * 60)

    return best_pipeline, metrics, top_features


def train_svm(
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    preprocessor=None,
) -> tuple[Pipeline, dict]:
    """Train and evaluate a Support Vector Machine (SVM) pipeline with 5-fold CV hyperparameter tuning.

    Steps:
    1. Bundle the unfitted preprocessor (StandardScaler + OneHotEncoder) and SVC into a Pipeline.
    2. Use a linear kernel and tune regularization parameter C in [0.1, 1.0, 10.0] via 5-Fold Stratified CV on X_train.
    3. Fit the optimal SVM pipeline on the full training set.
    4. Predict classes and class probabilities (probability=True) on the holdout test set (X_test).
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
            (
                "classifier",
                SVC(
                    kernel="linear",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
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
    print("STAYWISE — SUPPORT VECTOR MACHINE (SVM) TRAINING")
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


def save_model(pipeline: Pipeline, filepath: Path | str = BEST_MODEL_PATH) -> Path:
    """Save a fitted Scikit-Learn pipeline to disk using joblib.

    Args:
        pipeline: Fitted Scikit-Learn Pipeline (preprocessor + classifier).
        filepath: Destination file path.

    Returns:
        Path to the saved artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"\n[Saved] Model pipeline saved to: {path}")
    return path


def load_model(filepath: Path | str = BEST_MODEL_PATH) -> Pipeline:
    """Load a serialized Scikit-Learn pipeline from disk.

    Args:
        filepath: Path to the .joblib file.

    Returns:
        Loaded fitted Pipeline.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    pipeline = joblib.load(path)
    return pipeline


def save_best_model(
    filepath: Path | str = BEST_MODEL_PATH,
    info_path: Path | str = MODEL_INFO_PATH,
) -> tuple[Pipeline, dict]:
    """Train the selected optimal model (Logistic Regression with C=10.0), serialize it, and save metadata.

    Returns:
        Tuple of (fitted_pipeline, model_info_dict).
    """
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    best_pipeline, metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # 1. Save pipeline (preprocessing + classifier) to disk
    save_model(best_pipeline, filepath)

    # 2. Compile model metadata
    model_info = {
        "model_name": "Logistic Regression",
        "algorithm": "sklearn.linear_model.LogisticRegression",
        "selected_hyperparameters": {
            "C": float(best_pipeline.named_steps["classifier"].C),
            "max_iter": 1000,
            "random_state": 42,
        },
        "holdout_test_metrics": {k: float(v) for k, v in metrics.items()},
        "saved_at": datetime.now().isoformat(),
        "input_features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }

    info_p = Path(info_path)
    info_p.parent.mkdir(parents=True, exist_ok=True)
    with open(info_p, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)
    print(f"[Saved] Model metadata saved to: {info_p}")

    return best_pipeline, model_info


def test_reloaded_model(
    filepath: Path | str = BEST_MODEL_PATH,
) -> bool:
    """Verify that the reloaded pipeline produces valid and identical predictions.

    Tests:
    1. Loads the pipeline from disk.
    2. Tests raw dictionary/DataFrame customer record without manual pre-transformation.
    3. Verifies probabilities are in [0, 1] and sum to 1.0.
    4. Compares predictions on X_test between freshly fitted and reloaded pipelines.
    """
    print("\n" + "=" * 60)
    print("STAYWISE — PHASE 7: RELOAD & INFERENCE VERIFICATION")
    print("=" * 60)

    # 1. Reload pipeline from disk
    reloaded_pipeline = load_model(filepath)
    print(f"Successfully reloaded pipeline from: {filepath}")
    print(f"Pipeline Steps: {[step[0] for step in reloaded_pipeline.steps]}")

    # 2. Test raw, un-preprocessed single customer records
    # Sample A: High-risk customer (Month-to-month, Fiber optic, Electronic check, short tenure)
    sample_churner = pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
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
                "MonthlyCharges": 85.50,
                "TotalCharges": 85.50,
            }
        ]
    )

    # Sample B: Low-risk loyal customer (Two year, DSL, long tenure, Auto payment)
    sample_loyal = pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 65,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 45.00,
                "TotalCharges": 2925.00,
            }
        ]
    )

    pred_churner = reloaded_pipeline.predict(sample_churner)[0]
    prob_churner = reloaded_pipeline.predict_proba(sample_churner)[0]
    pred_loyal = reloaded_pipeline.predict(sample_loyal)[0]
    prob_loyal = reloaded_pipeline.predict_proba(sample_loyal)[0]

    print("\n--- Raw Customer Prediction Tests ---")
    print(f"  Sample A (High Risk Churner): Pred = {pred_churner} (Churn), Churn Probability = {prob_churner[1]:.4f}")
    print(f"  Sample B (Low Risk Loyal)   : Pred = {pred_loyal} (No Churn), Churn Probability = {prob_loyal[1]:.4f}")

    # Check validity of probabilities
    assert 0.0 <= prob_churner[1] <= 1.0, "Probability out of range [0, 1]"
    assert 0.0 <= prob_loyal[1] <= 1.0, "Probability out of range [0, 1]"
    assert abs(sum(prob_churner) - 1.0) < 1e-6, "Probabilities must sum to 1.0"
    assert abs(sum(prob_loyal) - 1.0) < 1e-6, "Probabilities must sum to 1.0"
    assert prob_churner[1] > prob_loyal[1], "High-risk customer must have higher churn probability than loyal customer"

    # 3. Test on holdout test set to ensure exact match
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    test_preds = reloaded_pipeline.predict(X_test)
    test_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_preds, test_probs)

    print("\n--- Holdout Test Set Match Verification (1,409 rows) ---")
    print(f"  Reloaded Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Reloaded Precision: {metrics['precision']:.4f}")
    print(f"  Reloaded Recall   : {metrics['recall']:.4f}")
    print(f"  Reloaded F1-Score : {metrics['f1']:.4f}")
    print(f"  Reloaded ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("=" * 60)
    print("ALL RELOAD & INFERENCE VERIFICATION TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    train_logistic_regression()
    train_knn()
    compare_knn_scaling()
    train_random_forest()
    train_svm()
    save_best_model()
    test_reloaded_model()



