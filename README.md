# StayWise

> **Explainable Customer Churn Prediction & Decision-Support System**  
> *Who will churn · Why · When (estimate) · What action to take · Is it worth it?*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Problem Statement

Telecom companies lose 15–25 % of customers annually to churn. Retaining a customer costs 5–7× less than acquiring a new one. Yet most churn tools produce only a score — leaving frontline teams unable to act. **StayWise** bridges that gap by combining classification, explainability, survival analysis, and a rule-based ROI engine into a single interactive dashboard.

---

## Project Objectives

| Objective | Mechanism |
|---|---|
| Predict churn probability | 4-model classification pipeline (LR, DT, RF, XGBoost) |
| Explain *why* a customer is at risk | SHAP values → plain-English reasons |
| Estimate *when* risk peaks | Kaplan-Meier survival analysis on tenure |
| Recommend a retention action | Rule engine mapping SHAP drivers → offers |
| Quantify ROI of action | Expected saving formula with configurable assumptions |

---

## Dataset

**IBM Telco Customer Churn**  
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Setup (download dataset)

1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click **Download**
3. Extract and place the CSV at:
   ```
   StayWise\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

### Schema (21 columns)

| Column | Type | Description |
|---|---|---|
| customerID | string | Row identifier (dropped before modeling) |
| gender | categorical | Male / Female |
| SeniorCitizen | integer | 0 or 1 |
| Partner | categorical | Yes / No |
| Dependents | categorical | Yes / No |
| tenure | integer | Months with company |
| PhoneService | categorical | Yes / No |
| MultipleLines | categorical | Yes / No / No phone service |
| InternetService | categorical | DSL / Fiber optic / No |
| OnlineSecurity | categorical | Yes / No / No internet service |
| OnlineBackup | categorical | Yes / No / No internet service |
| DeviceProtection | categorical | Yes / No / No internet service |
| TechSupport | categorical | Yes / No / No internet service |
| StreamingTV | categorical | Yes / No / No internet service |
| StreamingMovies | categorical | Yes / No / No internet service |
| Contract | categorical | Month-to-month / One year / Two year |
| PaperlessBilling | categorical | Yes / No |
| PaymentMethod | categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| MonthlyCharges | float | Monthly bill (USD) |
| TotalCharges | float | Cumulative charges (blank for new customers → imputed) |
| Churn | categorical → binary | Yes=1 / No=0 (target) |

### Known Limitations

- No `churn_date` column → survival "When" estimates use **tenure as a proxy** (clearly labelled throughout)
- No complaint or usage-volume columns → satisfaction signals are proxied via `TechSupport`, `OnlineSecurity`, `Contract`
- ROI assumptions (`offer_cost`, `success_rate`) are configurable defaults — treat as illustrative

---

## ML Workflow

```
Raw CSV
  └─► data.py: validate → clean → split (60/20/20 stratified)
        └─► features.py: engineer 4 new features
              └─► train.py: ColumnTransformer pipeline × 4 models
                    └─► evaluate.py: CV metrics, curves, model selection by PR-AUC
                          └─► Calibrate best model (isotonic)
                                └─► Save models/best_pipeline.joblib
                                      └─► explain.py / survival.py / roi.py / actions.py
                                            └─► app.py (Streamlit dashboard)
```

---

## Model Descriptions

| Model | Imbalance Handling | Notes |
|---|---|---|
| Logistic Regression | `class_weight='balanced'` | Scaled; good calibration baseline |
| Decision Tree | `class_weight='balanced'` | `entropy`, `max_depth=8`; interpretable |
| Random Forest | `class_weight='balanced'` | 300 trees; robust to noise |
| XGBoost | `scale_pos_weight` | 300 estimators; typically best PR-AUC |

---

## Evaluation Methodology

- **Train / Val / Test split**: 60 / 20 / 20 (stratified, `random_state=42`)
- **CV**: 5-fold stratified on training set
- **Model selection**: Highest mean val PR-AUC
- **Threshold selection**: Profit-maximising curve on **validation set** only
- **Test set**: Evaluated **once**, after all selection decisions are final

### Metrics Explained

| Metric | What it tells you |
|---|---|
| Accuracy | % of correct predictions (misleading for imbalanced classes) |
| Precision | Of predicted churners, how many actually churn |
| Recall | Of actual churners, how many we catch |
| F1-score | Harmonic mean of Precision and Recall |
| ROC-AUC | Ability to rank churners above non-churners |
| PR-AUC | Area under Precision-Recall curve; preferred for imbalanced datasets |

---

## Results

> ⚠️ **Placeholder** — run `python -m src.train` with the dataset to populate real numbers.

| Model | Val PR-AUC | Val ROC-AUC | Val F1 |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| **Best (Test set)** | **—** | **—** | **—** |

---

## Installation

### Prerequisites
- Python 3.11 (or compatible)
- pip

### Windows PowerShell

```powershell
# 1. Navigate to project
cd C:\Users\hp\OneDrive\Documents\StayWise

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset at:
#    data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv

# 5. Train models
python -m src.train

# 6. Launch dashboard
streamlit run app.py
```

---

## Commands Reference

| Task | Command |
|---|---|
| Install deps | `pip install -r requirements.txt` |
| Train models | `python -m src.train` |
| Run tests | `python -m pytest tests/ -v --tb=short` |
| Launch app | `streamlit run app.py` |
| Coverage report | `python -m pytest tests/ --cov=src --cov-report=term-missing` |

---

## Project Structure

```
StayWise/
├── app.py                   # Streamlit entry point
├── config.yaml              # All configuration
├── requirements.txt
├── Makefile
├── README.md
├── .gitignore
├── data/
│   ├── raw/                 # Place dataset here (gitignored)
│   └── processed/           # Cleaned data (gitignored)
├── models/                  # Saved pipelines (gitignored)
├── reports/                 # Metrics and plots (gitignored)
├── src/
│   ├── data.py              # Load, validate, clean, split
│   ├── features.py          # Feature engineering
│   ├── train.py             # Train 4 models, select best, save
│   ├── evaluate.py          # Metrics, curves, calibration
│   ├── predict.py           # Single and batch prediction API
│   ├── explain.py           # SHAP global and per-customer
│   ├── survival.py          # Kaplan-Meier churn windows
│   ├── roi.py               # ROI and expected saving
│   └── actions.py           # Rule engine: reason → offer
├── pages/
│   ├── 01_overview.py
│   ├── 02_customer_lookup.py
│   ├── 03_whatif_simulator.py
│   ├── 04_top_at_risk.py
│   └── 05_model_lab.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_features.py
    ├── test_train.py
    ├── test_predict.py
    ├── test_roi.py
    ├── test_actions.py
    └── test_batch.py
```

---

## Limitations

1. **No churn date** — "When" survival estimates use tenure as proxy. Results are directional only.
2. **No complaint/usage data** — Dissatisfaction is proxied by service subscriptions and contract type.
3. **ROI is illustrative** — `offer_cost` and `success_rate` defaults are editable in `config.yaml` and the UI.
4. **SHAP ≠ causation** — Feature importance explains the model, not the real-world causal mechanism.
5. **Static dataset** — No real-time ingestion; retrain when data changes.

---

## Future Improvements

- Real-time data pipeline (Kafka / Flink)
- Automated retraining schedule
- A/B test tracking for retention offers
- Multi-tenant deployment with auth
- Causal inference layer (DoWhy / EconML)

---

## Resume Bullet Suggestions

> ⚠️ Fill in the blanks `[X]` with actual measured values after training.

- *Built StayWise, an end-to-end explainable churn prediction system on the IBM Telco dataset (7,043 records, 26.5% churn rate), achieving test PR-AUC of [X] and ROC-AUC of [X] with a calibrated XGBoost pipeline*
- *Integrated SHAP-based per-customer explanations mapped to plain-English retention reasons and a rule-based offer engine with configurable ROI assumptions*
- *Implemented survival analysis (Kaplan-Meier) to estimate churn probability within 30/60/90-day windows from a customer's current tenure*
- *Delivered a 5-page Streamlit dashboard covering overview, single-customer lookup, what-if simulation, batch scoring, and model comparison with Plotly visualisations*

---

## Benchmarks (Published Results)

> These are cited literature results on the same dataset — **not** StayWise results.

| Reference | Model | ROC-AUC | Notes |
|---|---|---|---|
| [Kaggle top notebooks](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code) | XGBoost | ~0.85 | Varies by feature engineering |
| [Scikit-learn baseline](https://scikit-learn.org) | LR | ~0.78 | With standard preprocessing |

---

*StayWise — Built with Python, Scikit-learn, XGBoost, SHAP, lifelines, Streamlit, and Plotly*
