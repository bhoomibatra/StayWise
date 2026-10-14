"""Download the IBM Telco Customer Churn dataset from GitHub."""
import urllib.request
import os
import sys
import pandas as pd

URLS = [
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

out = r"C:\Users\hp\OneDrive\Documents\StayWise\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)

for url in URLS:
    try:
        print(f"Trying: {url}")
        urllib.request.urlretrieve(url, out)
        df = pd.read_csv(out)
        print(f"SUCCESS: {len(df)} rows x {len(df.columns)} cols")
        churn_col = "Churn" if "Churn" in df.columns else None
        if churn_col:
            dist = df[churn_col].value_counts().to_dict()
            print(f"Churn distribution: {dist}")
        print(f"First columns: {list(df.columns[:6])}")
        sys.exit(0)
    except Exception as e:
        print(f"  Failed: {e}")

print("All URLs failed.")
sys.exit(1)
