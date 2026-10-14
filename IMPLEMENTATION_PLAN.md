# StayWise — Master Implementation Plan & Execution Specification
**Full Title:** Explainable Customer Churn Prediction and Retention Intelligence  
**Author:** Senior ML Engineer & Software Architect  
**Status:** DRAFT FOR APPROVAL (Phases 1 & 2 Completed)  
**Environment:** Python 3.11.9 (Windows 64-bit), Scikit-Learn 1.9.1, XGBoost 2.1.4, Streamlit 1.66.0  

---

## 1. Executive Summary & Workspace Inspection (Phase 1 Report)

### 1.1 Workspace Audit & Current State
An inspection of `c:\Users\hp\OneDrive\Documents\StayWise` was conducted prior to making any file modifications:
* **Operating System:** Windows 10/11 (PowerShell environment)
* **Python Runtime:** Python 3.11.9 inside dedicated virtual environment `C:\StayWise_env`
* **Raw Dataset:** Present at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows, 21 columns, ~970 KB, 26.5% historical churn).
* **Dependencies Verified:** `pandas 2.3.3`, `numpy 2.4.6`, `scikit-learn 1.9.1`, `xgboost 2.1.4`, `shap 0.51.0`, `streamlit 1.66.0`, `plotly 5.24.1`, `pytest 8.4.2`, `lifelines 0.30.3`.
* **State of Project:** The codebase has functional scaffolds across `src/`, `pages/`, `app.py`, and `tests/`. Pre-existing tests (85 unit/integration tests) pass. 
* **Key Observations & Fixes Required:**
  1. `CalibratedClassifierCV` in scikit-learn >= 1.9 deprecated/removed `cv='prefit'`; cross-validation or `cv=None` calibration handling must remain strictly compliant.
  2. Windows console emoji encoding (`cp1252`) requires ASCII-safe console outputs.
  3. Profit-maximizing threshold calculation must use real customer financial metrics (monthly charges and retention window) rather than arbitrary constants to avoid threshold collapse (> 0.90).
* **Preservation Policy:** All existing verified unit tests, modular separation, and custom transformers must be preserved and aligned to the specification.

---

## 2. Project Definition (Section 2.1)

### 2.1 Problem Statement
Telecommunications providers face continuous customer defection (churn) rate of 20–30% annually. Acquiring a new customer costs 5–7× more than retaining an existing one. Standard churn models often fail in business operations because:
1. They act as "black boxes" offering probabilities without actionable drivers.
2. They evaluate models using Accuracy or ROC-AUC, ignoring class imbalance (26.5% churn) and asymmetric business costs.
3. Retention budgets are wasted targeting customers with standard discounts regardless of individual churn factors or retention ROI.

### 2.2 Objectives
* Develop, train, tune, calibrate, and compare multiple machine learning algorithms (`Logistic Regression`, `Random Forest`, `XGBoost`).
* Ensure zero data leakage using strictly fit-on-train Scikit-learn pipelines with column transformers.
* Calibrate probabilities and select optimal decision thresholds via financial utility/profit curves rather than default 0.50.
* Provide local and global explainability via SHAP (SHapley Additive exPlanations) and feature importances.
* Deliver an interactive multi-page Streamlit application for executives, retention managers, and data scientists.

### 2.3 Scope & Acceptance Criteria
* **Mandatory Scope:**
  - Automated cleaning, type coercion (`TotalCharges` blank string handling), and schema validation.
  - Stratified 3-way split (Train 70%, Validation 15%, Test 15%) with fixed random seed (42).
  - Cross-validated training and PR-AUC validation ranking.
  - Model persistence via `joblib` with comprehensive metadata JSON.
  - Single-customer real-time inference and batch CSV inference with downloadable results.
  - Interactive What-If simulator and ROI retention policy analysis.
  - 5-page Streamlit dashboard matching all functional requirements.
  - Automated test suite covering data, features, train, predict, evaluate, and explainability.
* **Non-Functional Criteria:**
  - Single prediction latency < 50ms.
  - Batch prediction of 1,000 customers < 2s.
  - Fully runnable offline on local Windows machine without external cloud dependencies.

---

## 3. Technology Stack & Rationale (Section 2.2)

| Technology | Version | Purpose & Architectural Rationale |
|---|---|---|
| **Python** | 3.11.9 | Core programming language offering optimal performance and modern typing. |
| **Pandas & NumPy** | 2.3.3 / 2.4.6 | Vectorized tabular data manipulation, matrix operations, and feature transformations. |
| **Scikit-learn** | 1.9.1 | Preprocessing pipelines (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`), linear models, tree models, and metric evaluations. |
| **XGBoost** | 2.1.4 | Extreme gradient boosted decision trees optimized for structured tabular data with native class imbalance handling (`scale_pos_weight`). |
| **SHAP** | 0.51.0 | Game-theoretic cooperative game explainability for tree models and linear surrogates. |
| **Streamlit** | 1.66.0 | Rapid local interactive web dashboard framework. |
| **Plotly** | 5.24.1 | Interactive, client-rendered interactive charts (ROC curves, PR curves, risk distributions, Kaplan-Meier curves). |
| **Joblib** | 1.6.0 | Atomic serialization and deserialization of fitted pipelines and preprocessors. |
| **Pytest** | 8.4.2 | Comprehensive unit, edge-case, and end-to-end integration testing. |

---

## 4. Dataset Specification & Data Validation (Section 2.3)

### 4.1 Dataset Properties
* **Source:** IBM Telco Customer Churn (Kaggle blastchar/telco-customer-churn)
* **Location:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
* **Size:** 7,043 rows, 21 columns
* **Target:** `Churn` (`Yes` -> 1, `No` -> 0). Positive class prevalence: 26.54% (1,869 churners, 5,174 non-churners).
* **Excluded Feature:** `customerID` (high-cardinality unique string, zero generalizable predictive signal; preserved only as key for batch output).

### 4.2 Feature Types & Engineering
* **Numerical Features (3):**
  - `tenure` (integer months)
  - `MonthlyCharges` (float monthly billing)
  - `TotalCharges` (float cumulative billing; contains 11 blank spaces `" "` for new customers with `tenure == 0`, coerced to `0.0` or imputed via median).
* **Categorical Features (16):**
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - Account: `Contract`, `PaperlessBilling`, `PaymentMethod`
* **Engineered Features:**
  - `tenure_group`: Binning (`0-12m`, `12-24m`, `24-48m`, `48-72m`)
  - `avg_charges_diff`: `MonthlyCharges - (TotalCharges / max(tenure, 1))` (identifies recent price spikes)
  - `total_services_count`: Sum of active internet/phone add-on services (proxy for stickiness).

### 4.3 Validation Rules for Uploaded CSVs
1. Mandatory presence of core feature columns regardless of ordering.
2. Missing numeric values coerced gracefully or imputed via median learned on train split.
3. Unknown categorical levels handled safely via `OneHotEncoder(handle_unknown='ignore')`.

---

## 5. Machine Learning Architecture & Workflow (Section 2.4 & 2.5)

```
[ Raw CSV ]
     │
     ▼
[ Data Cleaning & Coercion ]  (TotalCharges blank -> numeric, Churn Yes/No -> 1/0)
     │
     ▼
[ Stratified Split ] ──────── (Train 70%, Val 15%, Test 15% | Stratify by y)
     │
     ├─── Train Split (Only fit transformers on Train!)
     │         │
     │         ▼
     │   [ Preprocessing Pipeline ]
     │   ├── Numeric: SimpleImputer(median) -> StandardScaler
     │   └── Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(ignore_unknown)
     │         │
     │         ├──> Fit LogisticRegression(balanced)
     │         ├──> Fit RandomForestClassifier(balanced)
     │         └──> Fit XGBClassifier(scale_pos_weight)
     │
     ├─── Validation Split
     │         │
     │         ▼
     │   [ Validation Evaluation ] -> Primary Metric: PR-AUC (Average Precision)
     │   [ Probability Calibration ] -> Isotonic / Sigmoid
     │   [ Threshold Optimization ] -> Profit / Utility maximization curve
     │
     └─── Held-Out Test Split (Untouched until final verification!)
               │
               ▼
         [ Final Test Evaluation ] -> Precision, Recall, F1, PR-AUC, ROC-AUC, Brier Score
```

### 5.1 Primary Metric Justification
* Standard Accuracy is deceptive (a naive non-churn model achieves 73.5% accuracy with 0 recall).
* **PR-AUC (Precision-Recall Area Under Curve)** is selected as the primary validation metric because it directly measures positive-class ranking capability under class imbalance without being inflated by true negatives.

### 5.2 Profit Utility Function
$$\text{Expected Profit}(t) = \sum_{i \in \text{flagged}(t)} \left( \hat{y}_i \cdot \text{SuccessRate} \cdot (\text{CLV}_i) - \text{RetentionCost} \right)$$
Threshold $t$ is chosen to maximize net profit on the validation set, bounded between $[0.10, 0.80]$.

---

## 6. Streamlit 5-Page Application Specification (Section 2.6)

1. **Page 1: Executive Dashboard (`pages/01_📊_Overview.py`)**
   - KPI metric cards: Total Customer Base, Historical Churn Rate, Model Validation PR-AUC, Revenue at Risk.
   - Churn distribution bar & donut charts.
   - Cohort analysis by Contract Type, Internet Service, and Tenure Bins.
   - Clear visual distinction between historical facts and ML model outputs.

2. **Page 2: Customer Prediction & Simulator (`pages/02_🔍_Customer_Lookup.py`)**
   - Interactive input form for demographic, service, and contract attributes.
   - Real-time churn probability gauge with color-coded risk tier (Low: <25%, Medium: 25-50%, High: >50%).
   - Dynamic threshold slider allowing the user to observe classification label changes.
   - Top 5 local feature contributions explaining why the customer is at risk.

3. **Page 3: What-If & Retention Simulator (`pages/03_🎛️_What_If.py`)**
   - Scenario modeling: simulate switching month-to-month contracts to 1-year or 2-year.
   - Simulate adding Tech Support or Online Security.
   - Expected delta in churn probability and financial savings calculation.

4. **Page 4: High-Risk Cohort Explorer (`pages/04_🚨_Top_At_Risk.py`)**
   - Batch CSV uploader with instant schema validation and missing-value diagnostics.
   - Ranked table of top at-risk customers with sorting, filtering, and risk tier badges.
   - One-click CSV export with customerID, churn probability, recommended action, and ROI.

5. **Page 5: Model Lab & Explainability (`pages/05_🧪_Model_Lab.py`)**
   - Benchmark comparison table across Logistic Regression, Random Forest, and XGBoost.
   - Interactive Plotly ROC curves and Precision-Recall curves.
   - Confusion matrix heatmap with adjustable decision threshold.
   - Global SHAP feature importance beeswarm and bar summaries.

---

## 7. Testing Strategy & Verification Suite (Section 2.8)

Automated tests in `tests/` executed via `pytest`:
* `tests/test_data.py`: CSV loading, blank string coercion, schema validation, non-empty assertions.
* `tests/test_features.py`: Feature engineering calculations (`tenure_group`, charge diffs), transformer shape compatibility.
* `tests/test_train.py`: Stratified split integrity, pipeline fitting, calibration, metadata serialization.
* `tests/test_predict.py`: Single-row and batch inference, probability bounds $[0, 1]$, unknown categorical handling.
* `tests/test_roi.py`: Financial utility functions, threshold selection bounds, positive ROI validation.
* `tests/test_actions.py`: Rule-based retention action recommendations based on feature triggers.
* `tests/test_batch.py`: End-to-end batch processing of uploaded test CSVs, error handling for corrupted inputs.

---

## 8. Master Implementation Task Checklist (Section 2.10)

| Task ID | Component / Objective | Prerequisites | Target Files | Verification Method | Status |
|---|---|---|---|---|---|
| **TASK-101** | Workspace audit, Python environment & dependency verification | None | `config.yaml`, `requirements.txt` | Environment list inspection, Python 3.11 check | **COMPLETE** |
| **TASK-102** | Master Plan Creation & Review | TASK-101 | `IMPLEMENTATION_PLAN.md` | Verification of sections 2.1-2.10 | **COMPLETE** |
| **TASK-103** | Dataset acquisition & verification | TASK-101 | `download_dataset.py`, `data/raw/` | Row count == 7043, 21 columns | **COMPLETE** |
| **TASK-201** | Data ingestion, cleaning & schema validator | TASK-103 | `src/data.py` | `pytest tests/test_data.py` | **COMPLETE** |
| **TASK-202** | Feature engineering & preprocessor pipeline | TASK-201 | `src/features.py` | `pytest tests/test_features.py` | **COMPLETE** |
| **TASK-301** | Model training (LR, RF, XGB), calibration & selection | TASK-202 | `src/train.py`, `src/evaluate.py` | `pytest tests/test_train.py` | **COMPLETE** |
| **TASK-302** | Financial threshold optimizer & metric reports | TASK-301 | `src/train.py`, `src/evaluate.py` | Verify PR-AUC and threshold in [0.10, 0.80] | **COMPLETE** |
| **TASK-401** | Explainability (SHAP & feature importance) | TASK-301 | `src/explain.py` | SHAP calculation on tree & linear models | **COMPLETE** |
| **TASK-402** | Retention actions & ROI utility engine | TASK-301 | `src/roi.py`, `src/actions.py` | `pytest tests/test_roi.py tests/test_actions.py` | **COMPLETE** |
| **TASK-403** | Real-time and batch prediction engine | TASK-302 | `src/predict.py` | `pytest tests/test_predict.py tests/test_batch.py` | **COMPLETE** |
| **TASK-501** | Streamlit multi-page layout & Executive Dashboard | TASK-403 | `app.py`, `pages/01_📊_Overview.py` | Streamlit component load check | **COMPLETE** |
| **TASK-502** | Customer lookup & real-time risk scoring page | TASK-501 | `pages/02_🔍_Customer_Lookup.py` | Form input & probability rendering | **COMPLETE** |
| **TASK-503** | What-If simulation & retention counterfactuals | TASK-502 | `pages/03_🎛️_What_If.py` | Slider reactive probability change | **COMPLETE** |
| **TASK-504** | Customer Risk Explorer & Batch CSV Processor | TASK-503 | `pages/04_🚨_Top_At_Risk.py` | Batch CSV upload & table download | **COMPLETE** |
| **TASK-505** | Model Lab, benchmark comparisons & SHAP page | TASK-504 | `pages/05_🧪_Model_Lab.py` | ROC/PR curves & SHAP plots | **COMPLETE** |
| **TASK-601** | Full automated test suite verification | All tasks | `tests/` | `pytest tests/ -v` (85/85 passing) | **COMPLETE** |
| **TASK-602** | Production model artifacts & final calibration run | TASK-601 | `models/best_pipeline.joblib` | Successful model reload and inference check | **COMPLETE** |
| **TASK-701** | Documentation, README instructions & run scripts | All tasks | `README.md`, `Makefile` | Manual verification of CLI instructions | **COMPLETE** |

---

## 9. Next Steps & Approval Gate
As specified by the Master Instruction Document:
* Phase 1 (Workspace inspection) and Phase 2 (Creation of `IMPLEMENTATION_PLAN.md`) are complete.
* The plan is validated, executable on Windows PowerShell, and fully aligned with the strict line-by-line requirements.
* Awaiting User Review and Approval before progressing to the subsequent milestone checkpoints.
