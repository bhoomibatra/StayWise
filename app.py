"""StayWise — Customer Churn Prediction ML Lab Web Application.

An interactive Streamlit application demonstrating machine learning model
training, comparison, evaluation, and single-customer churn prediction.
"""

from pathlib import Path
import json
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_JSON_PATH = MODELS_DIR / "metrics.json"
PLOTS_PNG_PATH = MODELS_DIR / "evaluation_plots.png"


# Page Configuration
st.set_page_config(
    page_title="StayWise — Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_best_pipeline(filepath: Path = BEST_MODEL_PATH):
    """Load and cache the trained Scikit-Learn pipeline from disk."""
    if not filepath.exists():
        return None
    return joblib.load(filepath)


def load_metrics_data(filepath: Path = METRICS_JSON_PATH):
    """Load model evaluation metrics and feature importances from JSON."""
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Initialize session state for customer form inputs
def init_form_state():
    defaults = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
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
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_form_state()


def set_high_risk_preset():
    st.session_state.gender = "Female"
    st.session_state.SeniorCitizen = 0
    st.session_state.Partner = "No"
    st.session_state.Dependents = "No"
    st.session_state.tenure = 2
    st.session_state.PhoneService = "Yes"
    st.session_state.MultipleLines = "No"
    st.session_state.InternetService = "Fiber optic"
    st.session_state.OnlineSecurity = "No"
    st.session_state.OnlineBackup = "No"
    st.session_state.DeviceProtection = "No"
    st.session_state.TechSupport = "No"
    st.session_state.StreamingTV = "Yes"
    st.session_state.StreamingMovies = "Yes"
    st.session_state.Contract = "Month-to-month"
    st.session_state.PaperlessBilling = "Yes"
    st.session_state.PaymentMethod = "Electronic check"
    st.session_state.MonthlyCharges = 95.50
    st.session_state.TotalCharges = 191.00


def set_loyal_preset():
    st.session_state.gender = "Male"
    st.session_state.SeniorCitizen = 0
    st.session_state.Partner = "Yes"
    st.session_state.Dependents = "Yes"
    st.session_state.tenure = 68
    st.session_state.PhoneService = "Yes"
    st.session_state.MultipleLines = "Yes"
    st.session_state.InternetService = "DSL"
    st.session_state.OnlineSecurity = "Yes"
    st.session_state.OnlineBackup = "Yes"
    st.session_state.DeviceProtection = "Yes"
    st.session_state.TechSupport = "Yes"
    st.session_state.StreamingTV = "No"
    st.session_state.StreamingMovies = "No"
    st.session_state.Contract = "Two year"
    st.session_state.PaperlessBilling = "No"
    st.session_state.PaymentMethod = "Bank transfer (automatic)"
    st.session_state.MonthlyCharges = 45.00
    st.session_state.TotalCharges = 3060.00


def reset_form_defaults():
    st.session_state.gender = "Female"
    st.session_state.SeniorCitizen = 0
    st.session_state.Partner = "No"
    st.session_state.Dependents = "No"
    st.session_state.tenure = 12
    st.session_state.PhoneService = "Yes"
    st.session_state.MultipleLines = "No"
    st.session_state.InternetService = "Fiber optic"
    st.session_state.OnlineSecurity = "No"
    st.session_state.OnlineBackup = "No"
    st.session_state.DeviceProtection = "No"
    st.session_state.TechSupport = "No"
    st.session_state.StreamingTV = "Yes"
    st.session_state.StreamingMovies = "No"
    st.session_state.Contract = "Month-to-month"
    st.session_state.PaperlessBilling = "Yes"
    st.session_state.PaymentMethod = "Electronic check"
    st.session_state.MonthlyCharges = 75.00
    st.session_state.TotalCharges = 900.00


# Application Header
st.title("📡 StayWise — Customer Churn Prediction ML Lab")
st.markdown(
    "**An End-to-End Educational Machine Learning Laboratory Project**  \n"
    "Demonstrating exploratory analysis, zero-leakage preprocessing, 5-fold cross-validation, "
    "multi-algorithm comparison, and interactive single-customer inference."
)
st.divider()

# Navigation Tabs
tab_overview, tab_comparison, tab_predictor, tab_insights = st.tabs(
    [
        "📊 Project Overview",
        "⚖️ Model Comparison",
        "🔮 Customer Churn Predictor",
        "💡 Feature Insights & Viva Guide",
    ]
)


# ==============================================================================
# TAB 1: PROJECT OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("1. Problem Statement & Dataset Summary")
    st.markdown(
        """
        Customer churn represents the percentage of subscribers who discontinue their service. 
        In the telecommunications sector, acquiring a new subscriber is typically 5 to 7 times 
        more expensive than retaining an existing customer. **StayWise** demonstrates an end-to-end 
        machine learning pipeline that identifies at-risk subscribers early so retention teams can intervene.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "7,043 Customers")
    col2.metric("Feature Count", "20 Attributes")
    col3.metric("Churn Rate", "26.54% (Imbalanced)")
    col4.metric("Holdout Test Size", "1,409 (20% Split)")

    st.markdown("---")
    st.subheader("2. End-to-End ML Architecture")

    st.markdown(
        """
        ```
        Raw IBM Telco Dataset (7,043 rows, 21 columns)
          ├── 1. Data Cleaning: Fix 11 whitespace strings in TotalCharges -> Impute Median
          ├── 2. Stratified Train/Test Split (80% Train: 5,634 | 20% Test: 1,409)
          ├── 3. Scikit-Learn ColumnTransformer (Zero Data Leakage)
          │     ├── Numerical (tenure, MonthlyCharges, TotalCharges) -> Median Imputer + StandardScaler
          │     └── Categorical (16 features) -> Frequent Imputer + OneHotEncoder(ignore_unknown)
          ├── 4. 5-Fold Stratified Cross-Validation on Training Set (Hyperparameter Tuning)
          │     ├── Logistic Regression (Tuned C in [0.1, 1.0, 10.0])
          │     ├── K-Nearest Neighbors (Tuned odd k in [5, 9, 15, 21])
          │     ├── Random Forest (Tuned n_estimators in [50, 100], max_depth in [5, 10])
          │     └── Support Vector Machine (Tuned C in [0.1, 1.0, 10.0], linear kernel)
          ├── 5. Benchmark Evaluation on Untouched Holdout Test Set (Accuracy, Precision, Recall, F1, ROC-AUC)
          ├── 6. Pipeline Serialization with Joblib (`models/best_model.joblib`)
          └── 7. Web Application Serving via Streamlit
        ```
        """
    )

    st.markdown("---")
    st.subheader("3. Data Hygiene Note (The TotalCharges whitespace issue)")
    st.info(
        "**Exploration Finding:** 11 records in the dataset had `tenure = 0` and contained whitespace `' '` "
        "in the `TotalCharges` column, causing pandas to initially read it as an `object` string type. "
        "StayWise safely coerced these to `NaN` and imputed them using the median of the training split, "
        "completely preventing runtime crashes during modeling."
    )


# ==============================================================================
# TAB 2: MODEL COMPARISON
# ==============================================================================
with tab_comparison:
    st.subheader("1. Comparative Benchmark Across All 4 Algorithms")
    st.markdown(
        "All four models were evaluated on the **exact same untouched holdout test set** "
        "(1,409 rows, 26.54% churn) to ensure an authentic and fair benchmark."
    )

    metrics_data = load_metrics_data()

    if metrics_data and "metrics_summary" in metrics_data:
        summary_dict = metrics_data["metrics_summary"]
        table_rows = []
        for model_name, scores in summary_dict.items():
            table_rows.append(
                {
                    "Algorithm": model_name,
                    "Accuracy": f"{scores['accuracy'] * 100:.2f}%",
                    "Precision": f"{scores['precision'] * 100:.2f}%",
                    "Recall": f"{scores['recall'] * 100:.2f}%",
                    "F1-Score": f"{scores['f1']:.4f}",
                    "ROC-AUC": f"{scores['roc_auc']:.4f}",
                }
            )
        df_display = pd.DataFrame(table_rows).set_index("Algorithm")
        st.table(df_display)

        st.success(
            f"**Selected Deployment Model:** **{metrics_data.get('best_model', 'Logistic Regression')}**  \n"
            f"*{metrics_data.get('selection_rationale', '')}*"
        )
    else:
        st.warning("Model comparison metrics not found. Run `python src/evaluate.py` to generate `models/metrics.json`.")

    st.markdown("---")
    st.subheader("2. Visual Evaluation: Confusion Matrices & ROC Curves")

    if PLOTS_PNG_PATH.exists():
        st.image(str(PLOTS_PNG_PATH), caption="Confusion Matrices & Overlaid ROC Curves on Holdout Test Set", use_container_width=True)
    else:
        st.info("Evaluation plots not generated yet. Run `python src/evaluate.py` to produce `models/evaluation_plots.png`.")

    st.markdown("---")
    st.subheader("3. Viva Discussion: Why Accuracy Alone is Misleading")
    st.markdown(
        """
        - **The Majority Class Trap:** In an imbalanced dataset (~73.5% non-churn, ~26.5% churn), a dummy classifier that always predicts "No Churn" achieves **73.5% accuracy**, yet catches **0% of churners** (Recall = 0.0).
        - **Why Recall is Critical in Churn:**
          - **False Negative (FN):** Model says customer stays $\\rightarrow$ Customer cancels $\\rightarrow$ Company loses customer lifetime value (**High Cost**).
          - **False Positive (FP):** Model says customer churns $\\rightarrow$ Customer was staying $\\rightarrow$ Company sends a discount email (**Low Cost**).
          - Because False Negatives are far more damaging than False Positives, **Recall and F1-Score** are prioritized over raw Accuracy.
        """
    )


# ==============================================================================
# TAB 3: CUSTOMER CHURN PREDICTOR
# ==============================================================================
with tab_predictor:
    st.subheader("1. Interactive Customer Risk Predictor")
    st.markdown(
        "Enter subscriber attributes below, or click a preset profile for quick viva demonstration. "
        "The serialized Scikit-Learn pipeline will preprocess raw inputs and compute churn probability."
    )

    # Preset Profile Buttons
    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    with col_p1:
        st.button("🚨 Load High-Risk Profile", on_click=set_high_risk_preset, use_container_width=True)
    with col_p2:
        st.button("🛡️ Load Loyal Customer Profile", on_click=set_loyal_preset, use_container_width=True)
    with col_p3:
        st.button("🔄 Reset to Default Profile", on_click=reset_form_defaults, use_container_width=True)

    st.markdown("---")

    with st.form("churn_prediction_form"):
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown("##### 👤 Demographics & Tenure")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if st.session_state.gender == "Female" else 1)
            senior_val = st.selectbox("Senior Citizen", ["No (0)", "Yes (1)"], index=st.session_state.SeniorCitizen)
            senior_citizen = 1 if "Yes" in senior_val else 0
            partner = st.selectbox("Has Partner", ["No", "Yes"], index=0 if st.session_state.Partner == "No" else 1)
            dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=0 if st.session_state.Dependents == "No" else 1)
            tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=int(st.session_state.tenure))

        with col_c2:
            st.markdown("##### 🌐 Subscriptions & Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=0 if st.session_state.PhoneService == "No" else 1)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(st.session_state.MultipleLines))
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(st.session_state.InternetService))
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(st.session_state.OnlineSecurity))
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(st.session_state.OnlineBackup))
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(st.session_state.DeviceProtection))
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(st.session_state.TechSupport))
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(st.session_state.StreamingTV))
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(st.session_state.StreamingMovies))

        with col_c3:
            st.markdown("##### 💳 Contract & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(st.session_state.Contract))
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=0 if st.session_state.PaperlessBilling == "No" else 1)
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(st.session_state.PaymentMethod),
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(st.session_state.MonthlyCharges), step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(st.session_state.TotalCharges), step=10.0)

        predict_submitted = st.form_submit_button("🔮 Compute Churn Prediction", use_container_width=True)

    if predict_submitted:
        pipeline = load_best_pipeline()

        if pipeline is None:
            st.error("Error: Trained model pipeline (`models/best_model.joblib`) not found! Run Phase 7 to generate the model artifact.")
        else:
            # Build 1-row DataFrame with original column names
            customer_data = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": senior_citizen,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                    }
                ]
            )

            # Predict using the reloaded end-to-end pipeline
            churn_pred = pipeline.predict(customer_data)[0]
            churn_prob = pipeline.predict_proba(customer_data)[0][1]

            st.markdown("---")
            st.subheader("2. Prediction Result")

            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                st.metric("Predicted Outcome", "CHURN (Yes)" if churn_pred == 1 else "NO CHURN (Stay)")

            with col_res2:
                st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")

            with col_res3:
                if churn_prob > 0.60:
                    st.error("🚨 Risk Level: HIGH")
                elif churn_prob >= 0.35:
                    st.warning("⚠️ Risk Level: MEDIUM")
                else:
                    st.success("🛡️ Risk Level: LOW")

            st.progress(float(churn_prob))

            st.caption(
                "⚠️ **Disclaimer:** This output is a statistical prediction generated by a machine learning model "
                "based on historical Telco subscriber patterns, not a guarantee of future individual customer behavior."
            )


# ==============================================================================
# TAB 4: FEATURE INSIGHTS & VIVA GUIDE
# ==============================================================================
with tab_insights:
    st.subheader("1. Random Forest Feature Importance (MDI)")
    st.markdown(
        "Mean Decrease in Impurity (Gini Importance) indicates how useful each feature was across "
        "all decision trees in distinguishing churners from non-churners."
    )

    metrics_data = load_metrics_data()

    if metrics_data and "random_forest_features" in metrics_data:
        rf_feats = metrics_data["random_forest_features"]
        df_feat = pd.DataFrame(rf_feats)
        df_feat["importance_pct"] = df_feat["importance"] * 100

        # Clean display names
        df_feat["Feature Name"] = (
            df_feat["feature"]
            .str.replace("cat__", "")
            .str.replace("num__", "")
            .str.replace("_", ": ")
        )

        st.bar_chart(data=df_feat.set_index("Feature Name")["importance_pct"])

        st.table(
            df_feat[["Feature Name", "importance"]].rename(
                columns={"importance": "Gini Impurity Score"}
            )
        )

        st.warning(
            "⚠️ **Correlation vs. Causation:** Feature importance reflects mathematical predictive usefulness "
            "within tree splits. It does **not** prove that changing a feature directly causes customer retention."
        )
    else:
        st.info("Feature importance data not found in `models/metrics.json`.")

    st.markdown("---")
    st.subheader("2. Comprehensive ML Lab Viva Cheatsheet")

    with st.expander("Q1: Why do we serialize the Scikit-Learn Pipeline rather than just the classifier?"):
        st.markdown(
            "A raw classifier accepts only transformed numerical arrays matching the exact dimension "
            "it was trained on. A `Pipeline` bundles the `ColumnTransformer` (missing value imputation, "
            "categorical one-hot encoding, and feature scaling) together with the classifier, eliminating "
            "train-serving skew and allowing raw user inputs to be predicted directly."
        )

    with st.expander("Q2: Why is Feature Scaling mandatory for KNN but optional for Random Forest?"):
        st.markdown(
            "KNN relies on distance calculations (Euclidean metric: $\\sqrt{\\sum(x_i - y_i)^2}$). "
            "If `TotalCharges` spans $0 - 8000$ and `SeniorCitizen` is $0 - 1$, `TotalCharges` completely "
            "dominates distances without scaling. Random Forest splits nodes based only on ordering thresholds "
            "along individual features, making it mathematically invariant to monotonic feature scaling."
        )

    with st.expander("Q3: What is Data Leakage and how did StayWise prevent it?"):
        st.markdown(
            "Data leakage occurs when information from the test set influences model training (e.g. computing "
            "the global mean of `TotalCharges` on the whole dataset before splitting). StayWise performed a "
            "stratified train/test split **before** fitting any imputer, scaler, or encoder, ensuring transformers "
            "were fitted strictly on `X_train`."
        )

    with st.expander("Q4: Why was Logistic Regression selected over Random Forest and SVM for deployment?"):
        st.markdown(
            "In customer churn, minimizing False Negatives (catching subscribers who would leave) is paramount. "
            "Logistic Regression achieved a significantly higher Recall (**55.88%** vs. 42.78% for RF and 52.94% for SVM) "
            "and higher F1-score (**0.6040** vs. 0.5237 for RF and 0.5706 for SVM) on the holdout test set with top ROC-AUC (0.8411), "
            "while also offering fast, lightweight, and calibrated probability estimates."
        )

    with st.expander("Q5: What is Support Vector Machine (SVM) and how does it predict customer churn?"):
        st.markdown(
            "SVM is a supervised machine learning algorithm that finds the optimal decision boundary (hyperplane) "
            "separating churners ($y=1$) from non-churners ($y=0$) in high-dimensional feature space. "
            "For customer churn, it maps customer attributes (tenure, contract type, charges) into feature space "
            "and predicts churn based on which side of the hyperplane the customer lies."
        )

    with st.expander("Q6: What do the Hyperplane, Margin, and Support Vectors mean?"):
        st.markdown(
            "- **Hyperplane:** The linear decision boundary ($w^T x + b = 0$) that separates the two classes.\n"
            "- **Margin:** The geometric distance between the hyperplane and the closest data points of each class. "
            "SVM is a 'maximum margin classifier'—it maximizes this buffer to achieve good generalization.\n"
            "- **Support Vectors:** The critical data points closest to the hyperplane that define the margin. "
            "If other points far from the boundary move, the boundary does not change at all."
        )

    with st.expander("Q7: How does SVM differ from Logistic Regression, KNN, and Random Forest?"):
        st.markdown(
            "- **vs. Logistic Regression:** Logistic Regression models class posterior probabilities using a sigmoid function "
            "over all data points (log-loss), whereas SVM focuses only on the most difficult boundary cases (support vectors) "
            "using hinge loss.\n"
            "- **vs. KNN:** KNN is an instance-based lazy learner requiring pairwise distance calculation to all training points "
            "at query time. SVM explicitly learns boundary weights ($w, b$) during training and classifies quickly at test time.\n"
            "- **vs. Random Forest:** Random Forest is an ensemble of decision trees using orthogonal axis splits that capture complex "
            "step interactions without feature scaling. Linear SVM finds a single planar boundary and strictly requires feature scaling."
        )

st.divider()
st.caption("StayWise — Customer Churn Prediction ML Lab | Department of Computer Science & Engineering")
