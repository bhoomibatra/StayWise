# 📡 StayWise — Customer Churn Prediction and Retention Intelligence

**An end-to-end machine learning project for customer churn prediction, built as part of a college ML lab using Python and Scikit-learn.**

---

## 1. Project Overview

StayWise is an educational machine learning project that predicts whether a telecom customer is likely to cancel their subscription (churn). It walks through the complete ML pipeline — from raw data cleaning to an interactive web application — and compares four classification algorithms side by side.

The project was built to demonstrate core ML concepts clearly: preprocessing without data leakage, cross-validated hyperparameter tuning, fair model comparison on a holdout test set, and serving predictions through a Streamlit web interface.

---

## 2. Problem Statement

Customer churn — the loss of subscribers to a competitor or voluntary cancellation — is a major cost driver in the telecom industry. Acquiring a new customer typically costs five to seven times more than retaining an existing one. Identifying at-risk customers before they cancel allows the business to offer targeted discounts, contract upgrades, or proactive support, turning potential exits into long-term loyalty.

This project frames churn prediction as a **binary classification problem**: given a customer's account and service attributes, predict whether that customer will churn (`Yes`) or stay (`No`).

---

## 3. Objectives

1. Load and clean the IBM Telco Customer Churn dataset, handling the known `TotalCharges` whitespace issue.
2. Build a leak-free preprocessing pipeline using Scikit-learn's `ColumnTransformer`.
3. Train and evaluate four supervised classification algorithms:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest
   - Support Vector Machine (SVM)
4. Select the best model using meaningful metrics (Recall, F1-Score, ROC-AUC) rather than raw accuracy alone.
5. Serialize the complete pipeline (preprocessing + classifier) for reuse.
6. Build an interactive Streamlit web application for live single-customer churn predictions.

---

## 4. Features

- **Four ML algorithms** trained, tuned, and compared on the same holdout test set.
- **5-fold stratified cross-validation** on the training set for hyperparameter tuning — the test set is never touched during tuning.
- **Side-by-side confusion matrices** and **overlaid ROC curves** saved as `models/evaluation_plots.png`.
- **Feature importance ranking** from the Random Forest model displayed in the Streamlit app.
- **Interactive prediction form** in Streamlit — enter any customer's attributes and get an instant churn probability and risk level (Low / Medium / High).
- **Preset demo profiles** (High Risk Customer and Loyal Customer) for quick viva demonstrations.
- **Automated test suite** using pytest covering preprocessing integrity, all four model pipelines, pipeline reload, and edge-case inputs.

---

## 5. Dataset

**Source:** IBM Telco Customer Churn dataset  
**URL:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv  
**File location in project:** `data/telco_churn.csv`

| Property | Value |
|---|---|
| Total rows | 7,043 customers |
| Total columns | 21 (including target) |
| Target column | `Churn` — `Yes` (1) or `No` (0) |
| Churn rate | 26.54% (imbalanced — ~73.5% stay) |
| Dropped column | `customerID` (unique identifier, no predictive value) |

### Numerical Features (3)
| Feature | Description |
|---|---|
| `tenure` | Number of months the customer has been with the company (0–72) |
| `MonthlyCharges` | Current monthly bill amount ($18–$119) |
| `TotalCharges` | Cumulative charges over the customer's lifetime ($0–$8,685) |

> **Known Issue:** 11 rows with `tenure = 0` contain a whitespace string `" "` instead of a number in `TotalCharges`. These are detected and converted to `NaN` during loading, then imputed using the median of the training set.

### Categorical Features (16)
`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

---

## 6. Technology Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas ≥ 2.0 | Data loading and manipulation |
| numpy ≥ 1.24 | Numerical operations |
| scikit-learn ≥ 1.3 | Preprocessing, model training, evaluation, pipelines |
| matplotlib ≥ 3.7 | Confusion matrix and ROC curve plots |
| seaborn ≥ 0.12 | Plot styling |
| joblib ≥ 1.3 | Model serialization (saving and loading) |
| streamlit ≥ 1.30 | Interactive web application |
| pytest ≥ 7.4 | Automated unit testing |

---

## 7. Project Structure

```
StayWise/
├── data/
│   └── telco_churn.csv              # IBM Telco raw CSV (7,043 rows)
├── models/
│   ├── best_model.joblib            # Serialized Scikit-learn Pipeline (preprocessor + LR)
│   ├── metrics.json                 # Evaluation metrics for all 4 models + feature importances
│   ├── model_info.json              # Best model metadata and hyperparameters
│   └── evaluation_plots.png        # 4-panel confusion matrices + overlaid ROC curves
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # CSV loading, TotalCharges fix, EDA exploration
│   ├── preprocess.py                # ColumnTransformer, stratified train/test split
│   ├── train.py                     # Training functions: LR, KNN, RF, SVM, save/load
│   └── evaluate.py                  # Model comparison, confusion matrices, ROC curves
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py           # Unit tests: data loading, split integrity, NaN checks
│   └── test_model.py                # Unit tests: all 4 pipelines, reload, edge cases
├── app.py                           # Streamlit web application (single file)
├── requirements.txt                 # Pinned Python dependencies
├── IMPLEMENTATION_PLAN.md           # Phase-by-phase project execution plan
├── VIVA_PREPARATION.md              # 25+ viva Q&A for lab defense
└── README.md                        # This file
```

---

## 8. Data Preprocessing

All preprocessing is performed inside a Scikit-learn `ColumnTransformer` bundled into a `Pipeline`. This design guarantees **zero data leakage**: the preprocessor is fitted only on the training set and applied identically to the test set and any new input.

### Steps in Order

1. **Load raw CSV** — `src/data_loader.py` reads `data/telco_churn.csv`.
2. **Fix TotalCharges** — 11 whitespace strings (`" "`) are coerced to `NaN` using `pd.to_numeric(errors='coerce')`.
3. **Drop columns** — `customerID` is removed (unique identifier, no predictive signal). `Churn` is extracted as the binary target (`No` → 0, `Yes` → 1).
4. **Stratified 80/20 split** — `train_test_split(stratify=y, random_state=42)` produces:
   - Training set: **5,634 rows** (~26.54% churners)
   - Test set: **1,409 rows** (~26.54% churners)
5. **Numerical pipeline** — Applied to `tenure`, `MonthlyCharges`, `TotalCharges`:
   - `SimpleImputer(strategy='median')` — fills the 11 `NaN` values using the training median only.
   - `StandardScaler()` — standardizes to zero mean and unit variance.
6. **Categorical pipeline** — Applied to all 16 categorical features:
   - `SimpleImputer(strategy='most_frequent')` — fills any missing values.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` — creates binary dummy columns. Unknown categories at inference time are silently ignored rather than crashing.

---

## 9. Machine Learning Algorithms

### 9.1 Logistic Regression
Logistic Regression fits a linear combination of features and passes the result through the sigmoid function to produce a probability between 0 and 1. A customer is predicted as churner if this probability exceeds 0.5.

- **Hyperparameter tuned:** `C` (inverse regularization strength) — values tested: `[0.1, 1.0, 10.0]`
- **Best C found by 5-fold CV:** `C = 10.0`
- **Why it works well here:** The relationship between contract type, tenure, charges, and churn is approximately linear in log-odds space. Logistic Regression is fast, interpretable, and produces well-calibrated probabilities.

### 9.2 K-Nearest Neighbors (KNN)
KNN stores all training customers and at prediction time finds the `k` most similar customers by Euclidean distance, then predicts by majority vote. It has no explicit training phase — it is a "lazy learner".

- **Hyperparameter tuned:** `k` (number of neighbours) — values tested: `[5, 9, 15, 21]` (odd values to prevent tie votes)
- **Best k found by 5-fold CV:** `k = 21`
- **Feature scaling is mandatory:** Without `StandardScaler`, `TotalCharges` (range 0–8,685) would completely dominate the Euclidean distance calculation, making `SeniorCitizen` (range 0–1) invisible to the model.

### 9.3 Random Forest
Random Forest trains an ensemble of decision trees, each on a random bootstrap sample of the training data and a random subset of features at each split. Final predictions are made by majority vote across all trees (bagging).

- **Hyperparameters tuned:** `n_estimators` in `[50, 100]` and `max_depth` in `[5, 10]`
- **Best combination found by 5-fold CV:** `n_estimators=100`, `max_depth=5`
- **Feature scaling is not required:** Decision trees split on thresholds along individual feature axes, so their output is invariant to monotonic rescaling.
- **Feature importances** are extracted and displayed in the Streamlit app (Mean Decrease in Impurity / Gini Importance).

**Top 5 most important features from the trained model:**
1. `Contract_Month-to-month` — 16.61%
2. `tenure` — 14.19%
3. `OnlineSecurity_No` — 9.06%
4. `TotalCharges` — 7.75%
5. `InternetService_Fiber optic` — 7.42%

### 9.4 Support Vector Machine (SVM)
SVM finds the hyperplane that separates churners from non-churners with the **maximum margin** — the largest possible gap between the boundary and the nearest data points from either class. Only the data points on the margin boundaries (called **support vectors**) determine the boundary position.

- **Kernel:** `linear` (the boundary is a flat hyperplane in feature space)
- **Probability calibration:** `probability=True` enables Platt scaling so churn probabilities can be computed
- **Hyperparameter tuned:** `C` (penalty for margin violations) — values tested: `[0.1, 1.0, 10.0]`
- **Best C found by 5-fold CV:** `C = 0.1` (wider margin, stronger regularization)
- **Feature scaling is mandatory:** SVM maximizes a geometric margin, so features must be on a comparable scale.

---

## 10. Model Evaluation

All four models are evaluated on the **same untouched holdout test set** (1,409 rows, never seen during training or hyperparameter tuning).

### Metrics Explained

| Metric | What It Measures | Why It Matters for Churn |
|---|---|---|
| **Accuracy** | Proportion of all predictions that are correct | Misleading on imbalanced data — a model predicting "No Churn" for everyone scores 73.5% |
| **Precision** | Of all customers predicted to churn, what fraction actually did | Measures how often the churn alert is correct |
| **Recall** | Of all customers who actually churned, what fraction did the model catch | **Most critical metric** — missing a churner means losing them with no intervention |
| **F1-Score** | Harmonic mean of Precision and Recall | Balanced metric when both matter |
| **ROC-AUC** | Model's ability to rank churners above non-churners at all thresholds | Threshold-independent performance measure |

### Actual Results (Holdout Test Set: 1,409 rows)

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **80.55%** | 65.72% | **55.88%** | **0.6040** | **0.8411** |
| Random Forest | 79.35% | **67.51%** | 42.78% | 0.5237 | 0.8397 |
| Support Vector Machine | 78.85% | 61.88% | 52.94% | 0.5706 | 0.8266 |
| K-Nearest Neighbors | 78.00% | 58.94% | 56.42% | 0.5765 | 0.8272 |

### Confusion Matrix Values (SVM shown as example)

|  | Predicted: No Churn | Predicted: Churn |
|---|:---:|:---:|
| **Actual: No Churn** | 912 (True Negative) | 123 (False Positive) |
| **Actual: Churn** | 176 (False Negative) | 198 (True Positive) |

- **False Negative (176):** Model predicted the customer would stay, but they churned → **High cost** (lost customer)
- **False Positive (123):** Model predicted churn, but customer stayed → **Low cost** (wasted retention offer)

---

## 11. Best Model

**Selected model: Logistic Regression** (`C = 10.0`, `max_iter = 1000`)

Logistic Regression was selected because in customer churn prediction, **Recall is the most critical metric**: every missed churner represents a customer who leaves without any retention attempt. Logistic Regression achieves the highest Recall (55.88%), highest F1-score (0.6040), and highest ROC-AUC (0.8411) among all four models. It is also computationally lightweight and produces well-calibrated probabilities.

The saved pipeline file `models/best_model.joblib` contains the **full preprocessing + classifier chain** so that raw customer input (as a dictionary or DataFrame) can be fed directly and receive a prediction without any manual transformation.

---

## 12. Installation and Setup

### Prerequisites
- Python 3.10 or higher
- PowerShell (Windows) or a bash-compatible terminal

### Step-by-Step Setup

```powershell
# 1. Clone the repository (if not already on your machine)
git clone https://github.com/bhoomibatra/StayWise.git
cd StayWise

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Note:** `data/telco_churn.csv` is already included in the repository. If it is missing, the data loader will attempt to download it automatically from the IBM GitHub repository.

### Verify the Setup

```powershell
# Check that preprocessing works correctly
.\.venv\Scripts\python.exe src/preprocess.py

# Run the automated test suite (should show 9 passed)
.\.venv\Scripts\pytest.exe -v tests/
```

---

## 13. How to Use the Application

### Launch the Streamlit Web App

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open `http://localhost:8501` in your browser.

### Application Tabs

| Tab | Contents |
|---|---|
| 📊 **Project Overview** | Problem statement, dataset summary, ML architecture diagram |
| ⚖️ **Model Comparison** | 4-model benchmark table, confusion matrices, ROC curves |
| 🔮 **Customer Churn Predictor** | Input form for customer attributes, churn probability output, risk level |
| 💡 **Feature Insights & Viva Guide** | Feature importance chart, viva Q&A cheatsheet |

### Making a Prediction

1. Go to the **Customer Churn Predictor** tab.
2. Either use a **preset profile** (🚨 High Risk or 🛡️ Loyal Customer) or fill in the customer's details manually.
3. Click **"Compute Churn Prediction"**.
4. The app will display:
   - **Predicted Outcome:** CHURN (Yes) or NO CHURN (Stay)
   - **Churn Probability:** e.g., 67.2%
   - **Risk Level:** LOW (< 35%), MEDIUM (35–60%), or HIGH (> 60%)
   - A visual probability bar

### Retrain All Models (Optional)

Trained model artifacts already exist in `models/`. Only run these if you want to retrain from scratch:

```powershell
# Train all four models
.\.venv\Scripts\python.exe src/train.py

# Regenerate evaluation plots and metrics.json
.\.venv\Scripts\python.exe src/evaluate.py
```

---

## 14. Limitations and Future Improvements

### Current Limitations

1. **Class imbalance is not explicitly handled.** With ~73.5% non-churners and ~26.5% churners, all models struggle to achieve high Recall without also generating many false positives. Techniques like SMOTE oversampling, class-weight adjustment (`class_weight='balanced'`), or probability threshold tuning were not applied in this version.

2. **Linear SVM only.** The SVM is trained with a linear kernel. Non-linear kernels such as RBF (Radial Basis Function) might capture more complex churn patterns but would significantly increase training time on 5,634 training rows with 5-fold CV.

3. **No feature selection.** All 19 features (after dropping `customerID`) are used. Some features may be redundant or weakly predictive (e.g., `gender`). Applying recursive feature elimination or Lasso-based selection could improve generalisation.

4. **Static dataset.** The model is trained once on a snapshot of the IBM dataset. In a real production system, models would need to be periodically retrained as customer behaviour and market conditions change.

5. **No threshold tuning.** All models use the default 0.5 classification threshold. In high-recall scenarios, lowering the threshold (e.g., to 0.35) would catch more churners at the cost of more false positives.

### Possible Future Improvements

- Apply `class_weight='balanced'` to Logistic Regression and SVM to handle class imbalance.
- Test a non-linear SVM with RBF kernel and compare against the linear version.
- Try gradient boosting methods (XGBoost, LightGBM) for potential accuracy gains.
- Add SHAP values for per-customer prediction explainability in the Streamlit app.
- Build an automated retraining pipeline triggered by new data uploads.
- Add a customer lifetime value (CLV) calculation to prioritise intervention by expected revenue loss.

---

## 15. Project Status

| Phase | Description | Status |
|---|---|:---:|
| Phase 1 | Environment Setup & Dataset Exploration | ✅ Complete |
| Phase 2 | Data Preprocessing | ✅ Complete |
| Phase 3 | Logistic Regression | ✅ Complete |
| Phase 4 | K-Nearest Neighbors | ✅ Complete |
| Phase 5 | Random Forest | ✅ Complete |
| Phase 5B | Support Vector Machine | ✅ Complete |
| Phase 6 | Model Comparison & Evaluation | ✅ Complete |
| Phase 7 | Saving & Reloading the Best Model | ✅ Complete |
| Phase 8 | Streamlit Interface | ✅ Complete |
| Phase 9 | Testing & Final Verification | ✅ Complete |
| Phase 10 | Documentation & Viva Preparation | ✅ Complete |

---

*StayWise — Customer Churn Prediction ML Lab | Department of Computer Science & Engineering*
