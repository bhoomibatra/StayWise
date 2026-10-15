"""Data Preprocessing Module for StayWise ML Lab Project.

Constructs a leak-free Scikit-Learn ColumnTransformer pipeline for handling
missing values, scaling numerical features, and one-hot encoding categorical features.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data

# Feature definitions
TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def create_preprocessor() -> ColumnTransformer:
    """Create an unfitted Scikit-Learn ColumnTransformer preprocessor.

    - Numerical pipeline: Median imputation -> StandardScaler
    - Categorical pipeline: Most frequent imputation -> OneHotEncoder (handle_unknown='ignore')

    Returns:
        ColumnTransformer instance configured for the Telco dataset.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_data(
    data_path: Path = DEFAULT_DATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Load raw data, perform stratified train/test split, and return unfitted preprocessor.

    CRITICAL RULE (Zero Data Leakage):
    The preprocessor is returned unfitted. It will be fitted strictly on the
    training split (X_train) within downstream Scikit-Learn pipelines.

    Args:
        data_path: Location of the raw CSV file.
        test_size: Proportion of the dataset to allocate to the holdout test set (default: 0.20).
        random_state: Seed for reproducible random train/test splitting (default: 42).

    Returns:
        Tuple containing:
            - X_train: Raw training features DataFrame
            - X_test: Raw holdout test features DataFrame
            - y_train: Binary training target Series (0 = No, 1 = Yes)
            - y_test: Binary test target Series (0 = No, 1 = Yes)
            - preprocessor: Configured, unfitted ColumnTransformer
    """
    # 1. Load data with TotalCharges whitespace fix handled
    df = load_raw_data(data_path)

    # 2. Exclude customerID and target from feature set
    cols_to_drop = [TARGET_COLUMN] + [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=cols_to_drop)

    # 3. Map target to binary integer: 'No' -> 0, 'Yes' -> 1
    y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int)

    # 4. Stratified train/test split to preserve exact churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 5. Instantiate unfitted preprocessor
    preprocessor = create_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor


def get_transformed_feature_names(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> list[str]:
    """Fit preprocessor on X_train and retrieve human-readable output feature names.

    Args:
        preprocessor: ColumnTransformer preprocessor.
        X_train: Training DataFrame used to fit the transformers.

    Returns:
        List of generated feature names after one-hot encoding and scaling.
    """
    preprocessor.fit(X_train)
    return list(preprocessor.get_feature_names_out())


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep = prepare_data()
    print("=" * 60)
    print("STAYWISE — DATA PREPROCESSING VERIFICATION")
    print("=" * 60)
    print(f"X_train shape: {X_tr.shape} (Rows: {X_tr.shape[0]}, Features: {X_tr.shape[1]})")
    print(f"X_test shape : {X_te.shape} (Rows: {X_te.shape[0]}, Features: {X_te.shape[1]})")
    print(f"Train Churn Rate (Class 1): {y_tr.mean():.4f} ({y_tr.sum()} / {len(y_tr)})")
    print(f"Test Churn Rate  (Class 1): {y_te.mean():.4f} ({y_te.sum()} / {len(y_te)})")

    # Fit preprocessor strictly on train data to inspect transformed shape
    X_tr_transformed = prep.fit_transform(X_tr)
    X_te_transformed = prep.transform(X_te)
    feat_names = prep.get_feature_names_out()

    print(f"\nTransformed feature matrix shape: {X_tr_transformed.shape[1]} columns")
    print(f"First 5 transformed feature names: {list(feat_names[:5])}")
    print(f"Any NaN values in transformed train data? {pd.isna(X_tr_transformed).any()}")
    print(f"Any NaN values in transformed test data?  {pd.isna(X_te_transformed).any()}")
    print("=" * 60)
