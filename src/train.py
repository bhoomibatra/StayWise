"""
src/train.py
------------
Train 4 classification models on the Telco churn dataset, compare them
using stratified cross-validation, select the best by PR-AUC, calibrate
it, and save the full preprocessing + model pipeline via Joblib.

Usage
-----
    python -m src.train                  # uses config.yaml defaults
    python -m src.train --config my.yaml # custom config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None  # type: ignore

from src.data import clean, load_raw, split
from src.evaluate import compute_metrics
from src.features import engineer, get_feature_lists

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

def _load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ─── Preprocessing pipeline builder ──────────────────────────────────────────

def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes numeric columns with median and scales them.
    - Imputes categorical columns with most-frequent and one-hot encodes.

    Parameters
    ----------
    numeric_features : list[str]
    categorical_features : list[str]

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


# ─── Model definitions ────────────────────────────────────────────────────────

def _get_models(cfg: dict, scale_pos_weight: float) -> dict[str, Any]:
    """Return a dict of model name → unfitted estimator."""
    mc = cfg["models"]
    return {
        "Logistic Regression": LogisticRegression(
            C=mc["logistic_regression"]["C"],
            max_iter=mc["logistic_regression"]["max_iter"],
            class_weight=mc["logistic_regression"]["class_weight"],
            solver=mc["logistic_regression"]["solver"],
            random_state=cfg["data"]["random_state"],
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion=mc["decision_tree"]["criterion"],
            max_depth=mc["decision_tree"]["max_depth"],
            class_weight=mc["decision_tree"]["class_weight"],
            random_state=mc["decision_tree"]["random_state"],
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=mc["random_forest"]["n_estimators"],
            class_weight=mc["random_forest"]["class_weight"],
            random_state=mc["random_forest"]["random_state"],
            n_jobs=mc["random_forest"]["n_jobs"],
        ),
        "XGBoost": XGBClassifier(
            n_estimators=mc["xgboost"]["n_estimators"],
            learning_rate=mc["xgboost"]["learning_rate"],
            max_depth=mc["xgboost"]["max_depth"],
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=mc["xgboost"]["random_state"],
            verbosity=0,
        ),
    }


# ─── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    """
    Run stratified k-fold CV and return mean metrics.

    Returns
    -------
    dict with keys: cv_roc_auc, cv_pr_auc, cv_f1, cv_precision, cv_recall
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "average_precision", "f1", "precision", "recall"],
        return_train_score=False,
        n_jobs=-1,
    )
    return {
        "cv_roc_auc":   float(np.mean(scores["test_roc_auc"])),
        "cv_pr_auc":    float(np.mean(scores["test_average_precision"])),
        "cv_f1":        float(np.mean(scores["test_f1"])),
        "cv_precision": float(np.mean(scores["test_precision"])),
        "cv_recall":    float(np.mean(scores["test_recall"])),
    }


# ─── Main training function ────────────────────────────────────────────────────

def train(config_path: Path | None = None) -> dict:
    """
    Full training pipeline:
    1. Load and clean data.
    2. Engineer features.
    3. Split into train / val / test.
    4. Train 4 models with CV.
    5. Select best by val PR-AUC.
    6. Calibrate best model.
    7. Evaluate on test set (once).
    8. Save pipeline and comparison report.

    Returns
    -------
    dict with keys: best_model_name, val_metrics, test_metrics, comparison
    """
    cfg = _load_config(config_path)
    root = _project_root()
    seed = cfg["data"]["random_state"]
    n_folds = cfg["data"]["cv_folds"]
    models_dir = root / cfg["paths"]["models_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    logger.info("═══ Loading and cleaning data ═══")
    raw = load_raw()
    df = clean(raw)
    df = engineer(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split(df, config_path)

    feature_lists = get_feature_lists(X_train)
    numeric_feats = feature_lists["numeric"]
    categorical_feats = feature_lists["categorical"]
    logger.info("Features — numeric: %d | categorical: %d", len(numeric_feats), len(categorical_feats))

    # ── 2. Imbalance weight for XGBoost ───────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = round(neg / pos, 2)
    logger.info("Class imbalance — neg: %d | pos: %d | scale_pos_weight: %.2f", neg, pos, scale_pos_weight)

    # ── 3. Build preprocessor ─────────────────────────────────────────────────
    preprocessor = build_preprocessor(numeric_feats, categorical_feats)

    # ── 4. Train all models with CV ───────────────────────────────────────────
    logger.info("═══ Training and cross-validating 4 models ═══")
    models = _get_models(cfg, scale_pos_weight)
    comparison_rows: list[dict] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in models.items():
        logger.info("  Training: %s", model_name)
        t0 = time.time()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        # Cross-validate on training data only
        cv_metrics = cross_validate_model(pipeline, X_train, y_train, n_folds, seed)

        # Fit full pipeline on all training data
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Val metrics
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= cfg["business"]["threshold_default"]).astype(int)
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob, prefix="val_")

        row = {
            "model": model_name,
            "train_time_s": round(elapsed, 2),
            **cv_metrics,
            **val_metrics,
        }
        comparison_rows.append(row)
        fitted_pipelines[model_name] = pipeline

        logger.info(
            "    CV PR-AUC: %.4f | Val PR-AUC: %.4f | Val ROC-AUC: %.4f",
            cv_metrics["cv_pr_auc"], val_metrics["val_pr_auc"], val_metrics["val_roc_auc"],
        )

    # ── 5. Select best model by val PR-AUC ───────────────────────────────────
    selection_metric = cfg["training"]["selection_metric"]
    metric_key = f"val_{selection_metric}"
    best_row = max(comparison_rows, key=lambda r: r[metric_key])
    best_model_name = best_row["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    logger.info("═══ Best model: %s (val %s = %.4f) ═══", best_model_name, selection_metric, best_row[metric_key])

    # ── 6. Calibrate best model ───────────────────────────────────────────────
    logger.info("Calibrating best model with isotonic regression ...")
    calibrated_pipeline = _calibrate_pipeline(best_pipeline, X_train, y_train, cfg)

    # ── 7. Select threshold on validation set ─────────────────────────────────
    y_val_prob_cal = calibrated_pipeline.predict_proba(X_val)[:, 1]
    best_threshold = _select_threshold(y_val_prob_cal, y_val, cfg)
    logger.info("Selected classification threshold: %.3f (profit-maximising on val set)", best_threshold)

    # ── 8. Final evaluation on held-out test set (once) ───────────────────────
    logger.info("═══ Final evaluation on test set ═══")
    y_test_prob = calibrated_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, prefix="test_")
    logger.info(
        "Test — PR-AUC: %.4f | ROC-AUC: %.4f | F1: %.4f | Recall: %.4f",
        test_metrics["test_pr_auc"], test_metrics["test_roc_auc"],
        test_metrics["test_f1"], test_metrics["test_recall"],
    )

    # ── 9. Save artifacts ─────────────────────────────────────────────────────
    pipeline_path = root / cfg["paths"]["best_pipeline"]
    metadata = {
        "best_model_name": best_model_name,
        "threshold": best_threshold,
        "numeric_features": numeric_feats,
        "categorical_features": categorical_feats,
        "val_metrics": best_row,
        "test_metrics": test_metrics,
    }
    payload = {"pipeline": calibrated_pipeline, "metadata": metadata}
    joblib.dump(payload, pipeline_path)
    logger.info("Saved pipeline → %s", pipeline_path)

    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = root / cfg["paths"]["model_comparison"]
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Saved model comparison → %s", comparison_path)

    # Save metadata JSON
    meta_path = models_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved metadata → %s", meta_path)

    return {
        "best_model_name": best_model_name,
        "threshold": best_threshold,
        "val_metrics": best_row,
        "test_metrics": test_metrics,
        "comparison": comparison_rows,
    }


# ─── Calibration helper ───────────────────────────────────────────────────────

def _calibrate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict,
) -> CalibratedClassifierCV:
    """Wrap the fitted pipeline in isotonic calibration and refit.
    
    Note: sklearn 1.9 removed cv='prefit'. We use cv=None which fits
    calibration on the same training data (no cross-val for calibration).
    For production, use a dedicated calibration holdout set.
    """
    method = cfg["training"]["calibration_method"]
    # cv=None (default) uses cross-validated calibration on X_train
    # This is safe here since the estimator is already fitted
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


# ─── Threshold selection ──────────────────────────────────────────────────────

def _select_threshold(
    y_prob: np.ndarray,
    y_true: pd.Series,
    cfg: dict,
    avg_monthly: float = 65.0,
) -> float:
    """
    Select the classification threshold that maximises expected profit on the
    validation set. Returns the threshold in [0, 1], capped at 0.80.

    Profit per customer at threshold t:
        TP: success_rate x avg_customer_value - offer_cost
        FP: -offer_cost

    avg_monthly is a proxy for customer value when per-customer charges
    are unavailable during threshold selection.
    """
    offer_cost   = cfg["business"]["offer_cost"]
    success_rate = cfg["business"]["success_rate"]
    max_months   = cfg["business"]["max_remaining_months"]
    currency     = cfg["business"]["currency_conversion"]

    avg_cust_value = avg_monthly * currency * max_months
    tp_benefit = success_rate * avg_cust_value

    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = cfg["business"]["threshold_default"]
    best_profit = -np.inf

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        profit = tp * tp_benefit - (tp + fp) * offer_cost
        if profit > best_profit:
            best_profit = profit
            best_t = float(t)

    # Cap at 0.80 to ensure at least some churners are always flagged
    return round(min(best_t, 0.80), 3)


# ─── Testable entry point (accepts pre-split data) ───────────────────────────

def train_on_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    config_path: Path | None = None,
) -> dict:
    """
    Train pipeline on pre-split data. Used by tests to avoid file I/O.

    Returns
    -------
    dict with keys: pipeline, metadata
    """
    cfg = _load_config(config_path)
    seed = cfg["data"]["random_state"]
    n_folds = cfg["data"]["cv_folds"]

    feature_lists = get_feature_lists(X_train)
    numeric_feats = feature_lists["numeric"]
    categorical_feats = feature_lists["categorical"]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = round(neg / max(pos, 1), 2)

    preprocessor = build_preprocessor(numeric_feats, categorical_feats)
    models = _get_models(cfg, scale_pos_weight)

    best_pipeline = None
    best_val_pr = -1.0
    best_name = ""

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        from sklearn.metrics import average_precision_score
        val_pr = float(average_precision_score(y_val, y_val_prob))
        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_pipeline = pipeline
            best_name = model_name

    calibrated = _calibrate_pipeline(best_pipeline, X_train, y_train, cfg)
    y_val_prob_cal = calibrated.predict_proba(X_val)[:, 1]
    threshold = _select_threshold(y_val_prob_cal, y_val, cfg)

    y_test_prob = calibrated.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, prefix="test_")

    metadata = {
        "best_model_name": best_name,
        "threshold": threshold,
        "numeric_features": numeric_feats,
        "categorical_features": categorical_feats,
        "test_metrics": test_metrics,
    }
    return {"pipeline": calibrated, "metadata": metadata}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StayWise churn models")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    try:
        results = train(config_path=args.config)
        print("\nTraining complete!")
        print(f"   Best model  : {results['best_model_name']}")
        print(f"   Threshold   : {results['threshold']}")
        pr  = results['test_metrics'].get('test_pr_auc', 'N/A')
        roc = results['test_metrics'].get('test_roc_auc', 'N/A')
        f1  = results['test_metrics'].get('test_f1', 'N/A')
        rec = results['test_metrics'].get('test_recall', 'N/A')
        print(f"   Test PR-AUC : {pr:.4f}"  if isinstance(pr,  float) else f"   Test PR-AUC : {pr}")
        print(f"   Test ROC-AUC: {roc:.4f}" if isinstance(roc, float) else f"   Test ROC-AUC: {roc}")
        print(f"   Test F1     : {f1:.4f}"  if isinstance(f1,  float) else f"   Test F1     : {f1}")
        print(f"   Test Recall : {rec:.4f}" if isinstance(rec, float) else f"   Test Recall : {rec}")
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
