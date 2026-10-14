"""
pages/04_🚨_Top_At_Risk.py — StayWise Batch Scoring & Export
Upload a CSV, validate schema, score all rows, and download results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import pandas as pd
import streamlit as st

from src.data import REQUIRED_COLUMNS
from src.predict import load_pipeline, predict_batch, validate_batch_schema

st.set_page_config(page_title="Top At-Risk — StayWise", page_icon="🚨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);color:#e2e8f0;}
[data-testid="stSidebar"]{background:rgba(15,23,42,0.95);}
.sw-card{background:rgba(30,41,59,0.7);border:1px solid rgba(14,165,233,0.15);border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
</style>""", unsafe_allow_html=True)

st.markdown("## 🚨 Top At-Risk Customers")
st.caption("Upload a CSV of customers to score in batch, filter by risk, and export results")

@st.cache_resource
def _get_payload():
    try:
        return load_pipeline()
    except FileNotFoundError:
        return None

payload = _get_payload()
if payload is None:
    st.error("❌ No trained model. Train first in **🧪 Model Lab**.")
    st.stop()

threshold = payload["metadata"].get("threshold", 0.5)

# ─── Source selection ─────────────────────────────────────────────────────────
source = st.radio("Data source", ["Upload CSV", "Use training dataset"], horizontal=True)

df_input = None

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_input):,} rows")
    else:
        st.markdown("""
        <div class="sw-card">
            <p style="color:#94a3b8;">Upload a CSV with these columns (customerID is optional):</p>
            <code style="font-size:0.75rem;color:#0ea5e9;">gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges</code>
            <p style="color:#64748b;font-size:0.8rem;margin-top:0.5rem;">
            Note: A <code>Churn</code> column will be ignored if present — predictions are always generated from features only.
            </p>
        </div>""", unsafe_allow_html=True)
else:
    try:
        from src.data import clean, load_raw
        from src.features import engineer

        @st.cache_data(show_spinner="Loading dataset …")
        def _load_train():
            return engineer(clean(load_raw()))

        df_full = _load_train()
        df_input = df_full.drop(columns=["Churn"], errors="ignore")
        st.success(f"✅ Loaded full training dataset: {len(df_input):,} rows")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ─── Score ────────────────────────────────────────────────────────────────────
if df_input is not None:
    # Validate schema
    required_features = [c for c in REQUIRED_COLUMNS if c not in ("customerID", "Churn")]
    missing_cols = validate_batch_schema(df_input, required_features)

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    with st.spinner(f"Scoring {len(df_input):,} customers …"):
        results = predict_batch(df_input, payload=payload, threshold=threshold)

    # ── Filters ───────────────────────────────────────────────────────────────
    st.divider()
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        risk_filter = st.multiselect(
            "Filter by Risk Category",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with fc2:
        min_prob = st.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)
    with fc3:
        top_n = st.number_input("Show Top N", 10, len(results), min(100, len(results)))

    filtered = results[
        (results["risk_category"].isin(risk_filter)) &
        (results["churn_probability"] >= min_prob)
    ].sort_values("churn_probability", ascending=False).head(int(top_n))

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scored", f"{len(results):,}")
    m2.metric("Critical Risk", f"{(results['risk_category']=='Critical').sum():,}")
    m3.metric("High Risk", f"{(results['risk_category']=='High').sum():,}")
    m4.metric("Showing", f"{len(filtered):,}")

    # ── Color-coded table ─────────────────────────────────────────────────────
    st.markdown(f"### Results — Top {len(filtered)} customers")

    def _color_risk(val):
        colors = {"Critical": "#4c0519", "High": "#450a0a", "Medium": "#451a03", "Low": "#064e3b"}
        return f"background-color: {colors.get(val, '')}; color: white;"

    styled = filtered.style.applymap(_color_risk, subset=["risk_category"])
    styled = styled.format({"churn_probability": "{:.1%}"})
    st.dataframe(styled, use_container_width=True, height=400)

    # ── CSV export ────────────────────────────────────────────────────────────
    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="staywise_at_risk_customers.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    st.caption(
        "⚠️ Predictions are generated from feature columns only. "
        "Any 'Churn' column in your upload was ignored."
    )
