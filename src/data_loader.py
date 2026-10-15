"""Data Loader Module for StayWise ML Lab Project.

Handles downloading, loading, and exploratory data analysis of the
IBM Telco Customer Churn dataset.
"""

from pathlib import Path
import urllib.request
import pandas as pd

# Standard public repository URL for IBM Telco Customer Churn
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)

# Project root directory and default data location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"


def download_dataset(url: str = DATASET_URL, target_path: Path = DEFAULT_DATA_PATH) -> Path:
    """Download the Telco Churn CSV from a public repository if not already present.

    Args:
        url: Public URL to download the raw CSV from.
        target_path: Local filesystem path where the CSV should be saved.

    Returns:
        Path to the downloaded or existing CSV file.
    """
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading dataset from: {url}")
        # Add basic User-Agent header for urllib request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"Dataset successfully saved to: {target_path}")
        return target_path
    except Exception as exc:
        raise RuntimeError(
            f"Failed to automatically download dataset from {url}.\n"
            f"Error: {exc}\n"
            f"Manual fix: Download the Telco Churn CSV manually and save it to:\n"
            f"  {target_path.resolve()}"
        ) from exc


def load_raw_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Churn CSV and fix the whitespace issue in TotalCharges.

    Args:
        data_path: Path to the CSV file. If missing, attempts to download it.

    Returns:
        pd.DataFrame with 'TotalCharges' converted to numeric (float).
    """
    data_path = Path(data_path)
    if not data_path.exists():
        data_path = download_dataset(target_path=data_path)

    df = pd.read_csv(data_path)

    # In the raw IBM dataset, 11 rows have " " (single whitespace) in TotalCharges
    # converting with errors='coerce' replaces those spaces with NaN
    whitespace_count = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if whitespace_count > 0:
        print(f"Detected {whitespace_count} whitespace values in 'TotalCharges' converted to NaN.")

    return df


def explore_data(df: pd.DataFrame | None = None, data_path: Path = DEFAULT_DATA_PATH) -> dict:
    """Perform exploratory data analysis and print summary findings.

    Args:
        df: Optional pre-loaded DataFrame. If None, loads from data_path.
        data_path: Path to dataset if df is None.

    Returns:
        Dictionary containing EDA summary statistics.
    """
    if df is None:
        df = load_raw_data(data_path)

    total_rows, total_cols = df.shape
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    null_counts = df.isnull().sum()
    missing_cols = null_counts[null_counts > 0].to_dict()

    print("=" * 60)
    print("STAYWISE — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"Dataset Dimensions: {total_rows} rows, {total_cols} columns")
    print("\n--- Target Distribution ('Churn') ---")
    for label in churn_counts.index:
        count = churn_counts[label]
        pct = churn_pct[label]
        print(f"  {label:<5}: {count:,} ({pct:.2f}%)")

    print(f"\nPositive Class (Churners) Rate: {churn_pct.get('Yes', 0.0):.2f}%")
    print(f"Class Imbalance Ratio: ~{churn_pct.get('No', 0.0)/churn_pct.get('Yes', 1.0):.2f} : 1")

    print("\n--- Missing / Special Values ---")
    if missing_cols:
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing values ({count/total_rows*100:.3f}%)")
    else:
        print("  No missing values detected.")

    print("\n--- Numerical Feature Ranges ---")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in num_cols:
        if col in df.columns:
            print(f"  {col:<15}: min={df[col].min():.2f}, mean={df[col].mean():.2f}, max={df[col].max():.2f}")

    print("=" * 60)

    return {
        "rows": total_rows,
        "cols": total_cols,
        "churn_counts": churn_counts.to_dict(),
        "churn_pct": churn_pct.to_dict(),
        "missing_cols": missing_cols,
    }


if __name__ == "__main__":
    explore_data()
