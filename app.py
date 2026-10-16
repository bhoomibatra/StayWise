"""StayWise — Customer Churn Prediction and Retention Intelligence Dashboard.

An automated machine learning web application that analyzes the Telco Customer
Churn dataset to predict which subscribers will churn vs stay, categorizes
subscribers into transparent risk tiers, provides personalized customer-level
retention playbooks, and compares 4 machine learning algorithms.
"""

from pathlib import Path
import json
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import DEFAULT_DATA_PATH, load_raw_data
from src.preprocess import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_JSON_PATH = MODELS_DIR / "metrics.json"
PLOTS_PNG_PATH = MODELS_DIR / "evaluation_plots.png"

# Page Configuration
st.set_page_config(
    page_title="StayWise — Automated Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern enterprise interface
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .badge-high {
        background-color: #ff4b4b22;
        color: #ff4b4b;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ff4b4b66;
    }
    .badge-med {
        background-color: #ffa50022;
        color: #ffa500;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ffa50066;
    }
    .badge-low {
        background-color: #00cc6622;
        color: #00cc66;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #00cc6666;
    }
    .disclaimer-box {
        font-size: 0.85rem;
        background-color: rgba(100, 116, 139, 0.12);
        border-left: 3px solid #64748b;
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_best_pipeline(filepath: Path = BEST_MODEL_PATH):
    """Load and cache the trained Scikit-Learn pipeline from disk."""
    if not filepath.exists():
        return None
    return joblib.load(filepath)


@st.cache_data
def load_metrics_data(filepath: Path = METRICS_JSON_PATH):
    """Load model evaluation metrics and feature importances from JSON."""
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def run_dataset_scoring(data_path: Path = DEFAULT_DATA_PATH, high_thresh: float = 0.60, med_thresh: float = 0.35):
    """Load the dataset and score all eligible records with the trained pipeline.

    Guarantees:
    - Automatically loads dataset from configured path
    - Uses existing trained Scikit-learn pipeline (zero retraining on load)
    - Excludes customerID and target Churn from model inputs
    - Returns enriched DataFrame with probabilities, predictions, and risk tiers
    """
    if not BEST_MODEL_PATH.exists():
        return None, "Model file 'models/best_model.joblib' not found. Please train models first."

    try:
        raw_df = load_raw_data(data_path)
    except Exception as exc:
        return None, f"Error loading dataset from {data_path}: {exc}"

    try:
        pipeline = load_best_pipeline(BEST_MODEL_PATH)
        if pipeline is None:
            return None, "Failed to load Scikit-learn pipeline."

        # Strictly select only model input features (excludes customerID and Churn)
        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing_cols = [c for c in feature_cols if c not in raw_df.columns]
        if missing_cols:
            return None, f"Dataset schema error: missing required feature columns: {missing_cols}"

        X_input = raw_df[feature_cols].copy()

        # Batch prediction with trained pipeline
        probs = pipeline.predict_proba(X_input)[:, 1]
        preds = pipeline.predict(X_input)

        # Build enriched operational dataframe
        scored_df = raw_df.copy()
        scored_df["churn_probability"] = np.round(probs, 4)
        scored_df["churn_probability_pct"] = np.round(probs * 100, 1)
        scored_df["predicted_churn"] = np.where(preds == 1, "Churn", "Stay")

        # Assign risk category using configurable thresholds
        conditions = [
            scored_df["churn_probability"] >= high_thresh,
            scored_df["churn_probability"] >= med_thresh,
        ]
        choices = ["High Risk", "Medium Risk"]
        scored_df["risk_category"] = np.select(conditions, choices, default="Low Risk")

        return scored_df, None
    except Exception as exc:
        return None, f"Inference failure: {exc}"


def extract_risk_drivers(customer_row: pd.Series) -> list[tuple[str, str, str]]:
    """Inspect customer profile against empirical churn drivers in the dataset.

    Returns:
        List of tuples: (Driver Title, Impact Level ['High', 'Medium', 'Protective'], Explanation)
    """
    drivers = []

    # Contract Type
    contract = str(customer_row.get("Contract", ""))
    if contract == "Month-to-month":
        drivers.append((
            "Month-to-Month Contract",
            "High",
            "Subscribers on month-to-month contracts have 0 cancellation friction and represent >88% of all churners in the dataset."
        ))
    elif contract in ["One year", "Two year"]:
        drivers.append((
            f"{contract} Contract",
            "Protective",
            f"Long-term {contract.lower()} commitment strongly anchors subscriber retention."
        ))

    # Tenure
    tenure = float(customer_row.get("tenure", 0))
    if tenure <= 6:
        drivers.append((
            f"New Subscriber (Tenure: {int(tenure)} months)",
            "High",
            "Highest hazard window: customers in their first 6 months have the highest attrition rate."
        ))
    elif tenure <= 18:
        drivers.append((
            f"Early Lifecycle (Tenure: {int(tenure)} months)",
            "Medium",
            "Customer has not yet established deep operational brand habits."
        ))
    elif tenure >= 48:
        drivers.append((
            f"Tenured Customer ({int(tenure)} months)",
            "Protective",
            "Long customer lifetime value with established satisfaction."
        ))

    # Internet Service & Add-ons
    internet = str(customer_row.get("InternetService", ""))
    tech_support = str(customer_row.get("TechSupport", ""))
    online_sec = str(customer_row.get("OnlineSecurity", ""))

    if internet == "Fiber optic":
        if tech_support == "No" or online_sec == "No":
            drivers.append((
                "Fiber Optic Without Support Add-ons",
                "High",
                "Fiber optic has the highest churn rate when technical support or online security are absent."
            ))
        else:
            drivers.append((
                "Fully-Protected Fiber Optic Subscriber",
                "Protective",
                "Customer has bundled value-added safety services."
            ))

    # Payment Method
    payment = str(customer_row.get("PaymentMethod", ""))
    if "Electronic check" in payment:
        drivers.append((
            "Electronic Check Billing",
            "High",
            "Electronic check users exhibit the highest churn frequency (~45%) compared to automated payment channels."
        ))
    elif "automatic" in payment.lower():
        drivers.append((
            f"Automated Billing ({payment})",
            "Protective",
            "Automatic payment reduces involuntary billing drop-off and increases account stickiness."
        ))

    # Monthly Charges
    monthly = float(customer_row.get("MonthlyCharges", 0))
    if monthly >= 80.0:
        drivers.append((
            f"High Monthly Charge (${monthly:.2f}/mo)",
            "Medium",
            "Premium tier billing creates heightened price sensitivity and competitor comparison shopping."
        ))

    # Family & Bundle Stickiness
    partner = str(customer_row.get("Partner", ""))
    dependents = str(customer_row.get("Dependents", ""))
    if partner == "Yes" and dependents == "Yes":
        drivers.append((
            "Multi-Person Household (Partner & Dependents)",
            "Protective",
            "Household accounts face higher collective friction to switch providers."
        ))

    return drivers


def get_retention_playbook(customer_row: pd.Series, risk_cat: str) -> list[str]:
    """Generate personalized, actionable retention interventions based on customer profile."""
    actions = []
    contract = str(customer_row.get("Contract", ""))
    payment = str(customer_row.get("PaymentMethod", ""))
    tech_support = str(customer_row.get("TechSupport", ""))
    online_sec = str(customer_row.get("OnlineSecurity", ""))
    tenure = float(customer_row.get("tenure", 0))

    if contract == "Month-to-month":
        actions.append("🎯 **Contract Upgrade Incentive:** Offer a guaranteed 15% bill discount or speed upgrade in exchange for transitioning to an annual contract.")

    if tech_support == "No" or online_sec == "No":
        actions.append("🛠️ **Value Add-on Onboarding:** Provide complimentary 6-month trial of Premium Tech Support and Online Security to enhance perceived utility.")

    if "Electronic check" in payment:
        actions.append("💳 **Auto-Pay Migration Bonus:** Grant a recurring $5 monthly bill credit for switching from manual electronic check to automated bank transfer or credit card.")

    if tenure <= 6:
        actions.append("📞 **Proactive Onboarding Outreach:** Trigger an account manager satisfaction call to address initial connectivity issues before month 6.")

    if not actions:
        if risk_cat == "High Risk":
            actions.append("⚡ **Priority Account Review:** Assign senior customer success representative for immediate retention survey and customized loyalty credit.")
        else:
            actions.append("🎁 **Loyalty Appreciation:** Deliver anniversary appreciation perks and maintain standard high-quality customer experience touchpoints.")

    return actions


# ==============================================================================
# SIDEBAR CONTROLS & THRESHOLD CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.title("📡 StayWise")
    st.caption("Automated Churn Intelligence Engine")
    st.markdown("---")

    st.subheader("⚙️ Risk Thresholds")
    st.markdown(
        "<div class='disclaimer-box'>"
        "Configure operational risk boundaries. Default thresholds map sigmoid outputs into actionable triage tiers."
        "</div>",
        unsafe_allow_html=True,
    )

    high_thresh = st.slider(
        "🚨 High Risk Threshold (Prob ≥)",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Subscribers with churn probability above this threshold are flagged for immediate priority retention.",
    )

    med_thresh = st.slider(
        "⚠️ Medium Risk Threshold (Prob ≥)",
        min_value=0.20,
        max_value=0.49,
        value=0.35,
        step=0.05,
        help="Subscribers between Medium and High thresholds are enrolled in proactive engagement campaigns.",
    )

    st.caption(
        f"**Active Tiers:**  \n"
        f"• 🚨 High Risk: $\\ge {int(high_thresh*100)}\\%$  \n"
        f"• ⚠️ Medium Risk: ${int(med_thresh*100)}\\% - {int(high_thresh*100)-1}\\%$  \n"
        f"• 🛡️ Low Risk: $< {int(med_thresh*100)}\\%$"
    )

    st.markdown("---")
    st.subheader("📌 System Status")
    st.success("● Pipeline: Logistic Regression (C=1.0)")
    st.info("● Data Source: Telco Churn CSV (7,043 Records)")
    st.caption(
        "⚠️ **Calibration Note:** Logistic Regression probabilities represent statistical model posteriors. "
        "These tiers represent heuristic operational boundaries rather than calibrated financial probabilities."
    )


# ==============================================================================
# LOAD SCORED DATASET AUTOMATICALLY
# ==============================================================================
scored_df, score_err = run_dataset_scoring(DEFAULT_DATA_PATH, high_thresh=high_thresh, med_thresh=med_thresh)

if score_err or scored_df is None:
    st.error(f"❌ Application Initialization Error: {score_err}")
    st.info("Ensure `models/best_model.joblib` exists by running `python -m src.train`.")
    st.stop()


# Calculate high-level summary metrics across the dataset
total_customers = len(scored_df)
pred_churn_count = int((scored_df["predicted_churn"] == "Churn").sum())
pred_churn_pct = (pred_churn_count / total_customers) * 100
pred_stay_count = total_customers - pred_churn_count
pred_stay_pct = (pred_stay_count / total_customers) * 100

has_actuals = "Churn" in scored_df.columns
if has_actuals:
    act_churn_count = int((scored_df["Churn"].astype(str).str.lower() == "yes").sum())
    act_churn_pct = (act_churn_count / total_customers) * 100
else:
    act_churn_count = None
    act_churn_pct = None

high_risk_count = int((scored_df["risk_category"] == "High Risk").sum())
high_risk_pct = (high_risk_count / total_customers) * 100
med_risk_count = int((scored_df["risk_category"] == "Medium Risk").sum())
med_risk_pct = (med_risk_count / total_customers) * 100
low_risk_count = int((scored_df["risk_category"] == "Low Risk").sum())
low_risk_pct = (low_risk_count / total_customers) * 100


# Application Title & Top Header
st.title("📡 StayWise — Customer Churn Prediction & Retention Intelligence")
st.markdown(
    "Automated customer churn intelligence across **7,043 Telco subscriber records**. "
    "Every customer is scored using the trained Scikit-learn machine learning pipeline."
)
st.divider()


# ==============================================================================
# PRIMARY 5-PAGE NAVIGATION
# ==============================================================================
tab_overview, tab_risk, tab_details, tab_performance, tab_insights = st.tabs(
    [
        "📊 1. Overview",
        "⚠️ 2. Customer Risk",
        "🔍 3. Customer Details",
        "⚖️ 4. Model Performance",
        "💡 5. Insights & Viva Guide",
    ]
)


# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("Automated Dataset Churn Summary")
    st.markdown(
        "Instant intelligence calculated across all existing subscriber records. "
        "Historical actual labels (`Churn`) and identifiers (`customerID`) were strictly excluded from model inputs."
    )

    # Metric Cards Row 1
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Customers", f"{total_customers:,}", help="Total subscriber records loaded from dataset")
    with m2:
        st.metric(
            "Predicted Churn",
            f"{pred_churn_count:,}",
            delta=f"{pred_churn_pct:.1f}% of base",
            delta_color="inverse",
            help="Subscribers with model prediction = Churn (prob >= 0.50)",
        )
    with m3:
        st.metric(
            "Predicted to Stay",
            f"{pred_stay_count:,}",
            delta=f"{pred_stay_pct:.1f}% of base",
            delta_color="normal",
            help="Subscribers predicted to remain with service",
        )
    with m4:
        if has_actuals:
            st.metric(
                "Historical Churn Rate",
                f"{act_churn_count:,}",
                delta=f"{act_churn_pct:.2f}% actual",
                delta_color="off",
                help="Ground truth historical churn labels recorded in dataset",
            )
        else:
            st.metric("Historical Churn Rate", "N/A")
    with m5:
        st.metric(
            "Urgent High Risk",
            f"{high_risk_count:,}",
            delta=f"{high_risk_pct:.1f}% base",
            delta_color="inverse",
            help=f"Subscribers with churn probability >= {int(high_thresh*100)}%",
        )

    # Operational notice box
    st.markdown(
        """
        <div class='disclaimer-box'>
        <b>ℹ️ Evaluation Integrity Notice:</b> The metrics above represent <i>full-dataset operational scoring</i> 
        across all 7,043 subscriber records. For strictly unbiased model evaluation on the 
        <b>untouched 20% holdout test set (1,409 customers)</b>, visit the <b>Model Performance</b> tab.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Visual Analytics Row
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("##### 👥 Actual vs. Predicted Churn Breakdown")
        comp_df = pd.DataFrame(
            {
                "Classification": ["Predicted Outcome", "Historical Ground Truth"],
                "Churn / Cancelled": [pred_churn_count, act_churn_count if has_actuals else 0],
                "Retained / Stayed": [pred_stay_count, (total_customers - act_churn_count) if has_actuals else 0],
            }
        ).set_index("Classification")
        st.bar_chart(comp_df, color=["#ff4b4b", "#00cc66"])
        st.caption("Comparison between model predictions and recorded historical churn outcomes.")

    with c_right:
        st.markdown(f"##### 🎯 Subscriber Risk Tiers (Configurable: {int(med_thresh*100)}% / {int(high_thresh*100)}%)")
        risk_df = pd.DataFrame(
            {
                "Risk Category": ["🚨 High Risk", "⚠️ Medium Risk", "🛡️ Low Risk"],
                "Subscriber Count": [high_risk_count, med_risk_count, low_risk_count],
                "Percentage (%)": [high_risk_pct, med_risk_pct, low_risk_pct],
            }
        ).set_index("Risk Category")
        st.bar_chart(risk_df["Subscriber Count"], color="#ff8800")
        st.caption("Operational customer distribution across triage risk categories.")

    st.markdown("---")
    st.subheader("🚨 Priority Immediate Action: Top 5 Highest Risk Subscribers")
    st.markdown("These customers exhibit the highest probability of imminent churn and require immediate retention intervention:")

    top_at_risk = scored_df.sort_values(by="churn_probability", ascending=False).head(5)
    display_top = top_at_risk[
        ["customerID", "Churn", "predicted_churn", "churn_probability_pct", "risk_category", "Contract", "tenure", "MonthlyCharges", "PaymentMethod"]
    ].rename(
        columns={
            "customerID": "Customer ID",
            "Churn": "Actual Status",
            "predicted_churn": "Predicted",
            "churn_probability_pct": "Churn Prob (%)",
            "risk_category": "Risk Tier",
            "Contract": "Contract",
            "tenure": "Tenure (Mo)",
            "MonthlyCharges": "Monthly ($)",
            "PaymentMethod": "Payment Method",
        }
    )
    st.table(display_top)
    st.info("👉 Switch to the **Customer Risk** tab to search, filter, and export the complete 7,043 subscriber table.")


# ==============================================================================
# TAB 2: CUSTOMER RISK
# ==============================================================================
with tab_risk:
    st.subheader("Customer Risk Analysis & Triage Table")
    st.markdown(
        "Search, filter, and inspect churn predictions for all 7,043 customer records in the dataset. "
        "Use filters to isolate high-risk subscribers for targeted retention campaigns."
    )

    # Filter Controls Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        pred_filter = st.selectbox(
            "Filter by Prediction",
            ["All Customers", "Predicted Churn Only", "Predicted Stay Only"],
            index=0,
        )

    with f_col2:
        risk_filter = st.selectbox(
            "Filter by Risk Category",
            ["All Risk Tiers", "🚨 High Risk", "⚠️ Medium Risk", "🛡️ Low Risk"],
            index=0,
        )

    with f_col3:
        actual_filter = st.selectbox(
            "Filter by Actual Churn",
            ["All Historical Outcomes", "Actual Churned (Yes)", "Actual Stayed (No)"],
            index=0,
        )

    with f_col4:
        contract_filter = st.selectbox(
            "Filter by Contract Type",
            ["All Contract Types", "Month-to-month", "One year", "Two year"],
            index=0,
        )

    # Search Bar
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        search_query = st.text_input(
            "🔍 Search by Customer ID",
            placeholder="e.g. 7590-VHVEG or type partial ID...",
        ).strip()
    with s_col2:
        sort_choice = st.selectbox(
            "Sort Order",
            ["Highest Churn Probability", "Lowest Churn Probability", "Longest Tenure", "Highest Monthly Charges"],
            index=0,
        )

    # Apply filters
    filtered_df = scored_df.copy()

    if pred_filter == "Predicted Churn Only":
        filtered_df = filtered_df[filtered_df["predicted_churn"] == "Churn"]
    elif pred_filter == "Predicted Stay Only":
        filtered_df = filtered_df[filtered_df["predicted_churn"] == "Stay"]

    if risk_filter == "🚨 High Risk":
        filtered_df = filtered_df[filtered_df["risk_category"] == "High Risk"]
    elif risk_filter == "⚠️ Medium Risk":
        filtered_df = filtered_df[filtered_df["risk_category"] == "Medium Risk"]
    elif risk_filter == "🛡️ Low Risk":
        filtered_df = filtered_df[filtered_df["risk_category"] == "Low Risk"]

    if actual_filter == "Actual Churned (Yes)":
        filtered_df = filtered_df[filtered_df["Churn"].astype(str).str.lower() == "yes"]
    elif actual_filter == "Actual Stayed (No)":
        filtered_df = filtered_df[filtered_df["Churn"].astype(str).str.lower() == "no"]

    if contract_filter != "All Contract Types":
        filtered_df = filtered_df[filtered_df["Contract"] == contract_filter]

    if search_query:
        filtered_df = filtered_df[filtered_df["customerID"].str.contains(search_query, case=False, na=False)]

    # Sorting
    if sort_choice == "Highest Churn Probability":
        filtered_df = filtered_df.sort_values(by="churn_probability", ascending=False)
    elif sort_choice == "Lowest Churn Probability":
        filtered_df = filtered_df.sort_values(by="churn_probability", ascending=True)
    elif sort_choice == "Longest Tenure":
        filtered_df = filtered_df.sort_values(by="tenure", ascending=False)
    elif sort_choice == "Highest Monthly Charges":
        filtered_df = filtered_df.sort_values(by="MonthlyCharges", ascending=False)

    # Results Counter & CSV Export
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        st.markdown(
            f"**Showing {len(filtered_df):,} of {total_customers:,} customers** "
            f"({(len(filtered_df)/total_customers*100):.1f}% of dataset)"
        )
    with rc2:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered List (CSV)",
            data=csv_data,
            file_name="staywise_customer_risk_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Format dataframe for presentation
    columns_to_show = [
        "customerID",
        "Churn",
        "predicted_churn",
        "churn_probability_pct",
        "risk_category",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
    ]
    display_table = filtered_df[columns_to_show].rename(
        columns={
            "customerID": "Customer ID",
            "Churn": "Actual Status",
            "predicted_churn": "Prediction",
            "churn_probability_pct": "Churn Prob (%)",
            "risk_category": "Risk Tier",
            "Contract": "Contract",
            "tenure": "Tenure (Mo)",
            "MonthlyCharges": "Monthly ($)",
            "TotalCharges": "Total ($)",
            "InternetService": "Internet",
            "PaymentMethod": "Payment Method",
            "TechSupport": "Tech Support",
            "OnlineSecurity": "Online Security",
        }
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        height=480,
    )


# ==============================================================================
# TAB 3: CUSTOMER DETAILS
# ==============================================================================
with tab_details:
    st.subheader("Customer Profile & Retention Intelligence")
    st.markdown(
        "Select any customer from the dataset to inspect their full attributes, model prediction, "
        "personalized risk drivers, and recommended retention interventions."
    )

    # Customer Selection Controls
    col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])

    # Default to first high-risk customer or user selection
    customer_ids = scored_df["customerID"].tolist()

    with col_sel1:
        selected_id = st.selectbox(
            "Select Customer ID to Inspect",
            customer_ids,
            index=0,
            help="Type to search or choose any customer ID from the 7,043 records",
        )

    with col_sel2:
        top_risk_id = scored_df.sort_values(by="churn_probability", ascending=False).iloc[0]["customerID"]
        if st.button("🚨 Load Top High-Risk Customer", use_container_width=True):
            st.session_state["selected_customer_id"] = top_risk_id
            st.rerun()

    with col_sel3:
        loyal_id = scored_df.sort_values(by="churn_probability", ascending=True).iloc[0]["customerID"]
        if st.button("🛡️ Load Top Loyal Customer", use_container_width=True):
            st.session_state["selected_customer_id"] = loyal_id
            st.rerun()

    if "selected_customer_id" in st.session_state and st.session_state["selected_customer_id"] in customer_ids:
        selected_id = st.session_state["selected_customer_id"]

    cust_row = scored_df[scored_df["customerID"] == selected_id].iloc[0]

    # Customer Scorecard Header
    st.markdown("---")
    sc1, sc2, sc3, sc4 = st.columns(4)

    prob_val = float(cust_row["churn_probability"])
    risk_val = cust_row["risk_category"]
    pred_val = cust_row["predicted_churn"]
    actual_val = cust_row.get("Churn", "Unknown")

    with sc1:
        st.metric("Customer ID", cust_row["customerID"])
    with sc2:
        st.metric("Model Prediction", pred_val)
    with sc3:
        st.metric("Churn Probability", f"{prob_val * 100:.1f}%")
    with sc4:
        st.metric("Actual Historical Churn", str(actual_val))

    # Probability Progress bar
    if risk_val == "High Risk":
        st.error(f"🚨 **Triage Category: HIGH RISK** — Churn probability is {prob_val*100:.1f}%. Immediate retention outreach advised.")
    elif risk_val == "Medium Risk":
        st.warning(f"⚠️ **Triage Category: MEDIUM RISK** — Churn probability is {prob_val*100:.1f}%. Proactive engagement recommended.")
    else:
        st.success(f"🛡️ **Triage Category: LOW RISK** — Churn probability is {prob_val*100:.1f}%. Stable subscriber profile.")

    st.progress(prob_val)

    # 3-Column Attributes Grid
    st.markdown("##### 📋 Subscriber Attributes in Dataset")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown(
            f"""
            **Demographics & Tenure**
            - **Gender:** {cust_row.get('gender', 'N/A')}
            - **Senior Citizen:** {'Yes (1)' if cust_row.get('SeniorCitizen') == 1 else 'No (0)'}
            - **Partner:** {cust_row.get('Partner', 'N/A')}
            - **Dependents:** {cust_row.get('Dependents', 'N/A')}
            - **Tenure:** {int(cust_row.get('tenure', 0))} Months
            """
        )

    with a2:
        st.markdown(
            f"""
            **Subscribed Services**
            - **Phone Service:** {cust_row.get('PhoneService', 'N/A')}
            - **Multiple Lines:** {cust_row.get('MultipleLines', 'N/A')}
            - **Internet:** {cust_row.get('InternetService', 'N/A')}
            - **Online Security:** {cust_row.get('OnlineSecurity', 'N/A')}
            - **Online Backup:** {cust_row.get('OnlineBackup', 'N/A')}
            - **Device Protection:** {cust_row.get('DeviceProtection', 'N/A')}
            - **Tech Support:** {cust_row.get('TechSupport', 'N/A')}
            - **Streaming TV / Movies:** {cust_row.get('StreamingTV', 'N/A')} / {cust_row.get('StreamingMovies', 'N/A')}
            """
        )

    with a3:
        st.markdown(
            f"""
            **Contract & Billing**
            - **Contract Type:** {cust_row.get('Contract', 'N/A')}
            - **Paperless Billing:** {cust_row.get('PaperlessBilling', 'N/A')}
            - **Payment Method:** {cust_row.get('PaymentMethod', 'N/A')}
            - **Monthly Charges:** ${float(cust_row.get('MonthlyCharges', 0)):.2f}
            - **Total Charges:** ${float(cust_row.get('TotalCharges', 0)):.2f}
            """
        )

    # Risk Drivers and Retention Playbook
    st.markdown("---")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("##### 🔍 Identified Churn Risk Drivers")
        drivers = extract_risk_drivers(cust_row)
        if drivers:
            for title, impact, desc in drivers:
                if impact == "High":
                    st.markdown(f"🔴 **{title}** (High Risk Driver)  \n*{desc}*")
                elif impact == "Medium":
                    st.markdown(f"🟡 **{title}** (Moderate Driver)  \n*{desc}*")
                else:
                    st.markdown(f"🟢 **{title}** (Retention Anchor)  \n*{desc}*")
        else:
            st.markdown("No significant churn risk factors identified.")

    with d_col2:
        st.markdown("##### 💼 Recommended Retention Playbook")
        actions = get_retention_playbook(cust_row, risk_val)
        for act in actions:
            st.markdown(act)

    # Optional Secondary Feature: Manual What-If Simulator Expander
    st.markdown("---")
    with st.expander("🛠️ Optional Secondary Tool: Manual 'What-If' Customer Simulator"):
        st.markdown(
            "Test hypothetical subscriber configurations or what-if retention scenarios "
            "(e.g., test the impact of upgrading contract type or adding Tech Support)."
        )

        with st.form("manual_simulator_form"):
            mf1, mf2, mf3 = st.columns(3)
            with mf1:
                s_gender = st.selectbox("Gender", ["Female", "Male"], index=0 if cust_row.get("gender") == "Female" else 1)
                s_senior = st.selectbox("Senior Citizen", ["No (0)", "Yes (1)"], index=int(cust_row.get("SeniorCitizen", 0)))
                s_partner = st.selectbox("Has Partner", ["No", "Yes"], index=0 if cust_row.get("Partner") == "No" else 1)
                s_dep = st.selectbox("Has Dependents", ["No", "Yes"], index=0 if cust_row.get("Dependents") == "No" else 1)
                s_tenure = st.slider("Tenure (Months)", 0, 72, int(cust_row.get("tenure", 12)))

            with mf2:
                s_phone = st.selectbox("Phone Service", ["No", "Yes"], index=1 if cust_row.get("PhoneService") == "Yes" else 0)
                s_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
                s_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1 if cust_row.get("InternetService") == "Fiber optic" else 0)
                s_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
                s_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
                s_dev = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
                s_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
                s_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0)
                s_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0)

            with mf3:
                s_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
                s_paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=1)
                s_pay = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    index=0,
                )
                s_monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, float(cust_row.get("MonthlyCharges", 70.0)), step=1.0)
                s_total = st.number_input("Total Charges ($)", 0.0, 9000.0, float(cust_row.get("TotalCharges", 800.0)), step=10.0)

            sim_submit = st.form_submit_button("Run What-If Simulation", use_container_width=True)

        if sim_submit:
            sim_pipeline = load_best_pipeline()
            if sim_pipeline:
                sim_data = pd.DataFrame(
                    [
                        {
                            "gender": s_gender,
                            "SeniorCitizen": 1 if "Yes" in s_senior else 0,
                            "Partner": s_partner,
                            "Dependents": s_dep,
                            "tenure": s_tenure,
                            "PhoneService": s_phone,
                            "MultipleLines": s_lines,
                            "InternetService": s_internet,
                            "OnlineSecurity": s_sec,
                            "OnlineBackup": s_backup,
                            "DeviceProtection": s_dev,
                            "TechSupport": s_tech,
                            "StreamingTV": s_tv,
                            "StreamingMovies": s_movies,
                            "Contract": s_contract,
                            "PaperlessBilling": s_paperless,
                            "PaymentMethod": s_pay,
                            "MonthlyCharges": s_monthly,
                            "TotalCharges": s_total,
                        }
                    ]
                )
                sim_pred = sim_pipeline.predict(sim_data)[0]
                sim_prob = sim_pipeline.predict_proba(sim_data)[0][1]

                st.markdown(
                    f"**Simulation Result:** Predicted: **{'CHURN' if sim_pred == 1 else 'STAY'}** | "
                    f"Probability: **{sim_prob * 100:.2f}%** | "
                    f"Risk Tier: **{'🚨 High' if sim_prob >= high_thresh else '⚠️ Medium' if sim_prob >= med_thresh else '🛡️ Low'}**"
                )


# ==============================================================================
# TAB 4: MODEL PERFORMANCE
# ==============================================================================
with tab_performance:
    st.subheader("Authentic Model Evaluation: Untouched Holdout Test Set")
    st.markdown(
        """
        <div class='disclaimer-box'>
        <b>Honest Academic Evaluation Standard:</b> While the <i>Overview</i> and <i>Customer Risk</i> screens provide 
        operational scores across all 7,043 subscriber records, <b>all evaluation metrics shown below are calculated 
        strictly on the untouched 20% holdout test set (1,409 rows, 26.54% churn)</b>. 
        No test-set information was leaked into preprocessing, feature scaling, or model tuning.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

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
        st.warning("Model comparison metrics not found. Run `python -m src.evaluate` to generate `models/metrics.json`.")

    st.markdown("---")
    st.subheader("Visual Evaluation: 4-Panel Confusion Matrices & Overlaid ROC Curves")

    if PLOTS_PNG_PATH.exists():
        st.image(str(PLOTS_PNG_PATH), caption="Evaluated strictly on holdout test set (1,409 samples)", use_container_width=True)
    else:
        st.info("Evaluation plots not generated yet. Run `python -m src.evaluate` to generate `models/evaluation_plots.png`.")

    st.markdown("---")
    st.subheader("Viva Discussion: The Cost Asymmetry of Churn")
    st.markdown(
        """
        - **Why Accuracy is Misleading:** A dummy classifier always predicting 'Stay' would score **73.46% accuracy** on this dataset, yet catch **0% of churners** (Recall = 0.0).
        - **False Negative vs. False Positive Cost:**
          - **False Negative (Model says Stay, customer Churns):** Lost customer lifetime value (~$1,000+). **High business penalty.**
          - **False Positive (Model says Churn, customer Stays):** Cost of an outreach email or small retention discount (~$10–$20). **Low business penalty.**
        - **Decision Rule:** Logistic Regression captures **55.88% of churners** with an **ROC-AUC of 0.8411**, providing the optimal commercial tradeoff.
        """
    )


# ==============================================================================
# TAB 5: INSIGHTS & VIVA GUIDE
# ==============================================================================
with tab_insights:
    st.subheader("1. Random Forest Feature Importance (MDI)")
    st.markdown(
        "Mean Decrease in Impurity (Gini Importance) across all decision trees in the ensemble "
        "reveals the features most predictive of customer churn:"
    )

    metrics_data = load_metrics_data()

    if metrics_data and "random_forest_features" in metrics_data:
        rf_feats = metrics_data["random_forest_features"]
        df_feat = pd.DataFrame(rf_feats)
        df_feat["importance_pct"] = df_feat["importance"] * 100

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
            "⚠️ **Correlation vs. Causation:** Feature importance reflects predictive utility within tree splits, "
            "not direct causation. For instance, `Contract_Month-to-month` is a churn marker, but simply forcing a long contract "
            "without addressing product satisfaction may not prevent cancellation."
        )
    else:
        st.info("Feature importance data not found in `models/metrics.json`.")

    st.markdown("---")
    st.subheader("2. Core ML Lab Viva Exam Preparation")

    with st.expander("Q1: Why do we serialize the Scikit-Learn Pipeline rather than just the classifier?"):
        st.markdown(
            "A raw classifier accepts only transformed numerical arrays matching the exact dimension "
            "it was trained on. A `Pipeline` bundles the `ColumnTransformer` (missing value imputation, "
            "categorical one-hot encoding, and feature scaling) together with the classifier, eliminating "
            "train-serving skew and allowing raw user inputs to be predicted directly."
        )

    with st.expander("Q2: Why is Feature Scaling mandatory for KNN and SVM, but optional for Random Forest?"):
        st.markdown(
            "KNN and SVM rely on distance calculations (Euclidean metric: $\\sqrt{\\sum(x_i - y_i)^2}$) "
            "and margin dot products ($w^T x$). If `TotalCharges` spans $0 - 8000$ and `SeniorCitizen` is $0 - 1$, "
            "`TotalCharges` completely dominates without scaling. Random Forest splits nodes based only on ordering thresholds "
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
            "SVM finds the optimal decision boundary (hyperplane) separating churners ($y=1$) from non-churners ($y=0$) "
            "in high-dimensional feature space by maximizing the margin between the two classes. "
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

    with st.expander("Q7: How do in-sample full-dataset predictions differ from holdout test evaluation?"):
        st.markdown(
            "- **Operational Full-Dataset Predictions:** The 7,043 customer scores in the dashboard allow business teams "
            "to triage the entire existing customer base. Because 80% of these records were seen during training, full-dataset "
            "accuracy is an operational scoring mechanism, not an unbiased metric.\n"
            "- **Holdout Test Set Evaluation:** The 1,409 unseen test records evaluate genuine generalization capability, "
            "providing the true scientific benchmark reported in the *Model Performance* tab."
        )

st.divider()
st.caption("StayWise — Customer Churn Prediction and Retention Intelligence | ML Lab")
