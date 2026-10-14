"""
pages/02_🔍_Customer_Lookup.py — StayWise Customer Lookup
Single-customer churn prediction with explanation, survival windows, and ROI.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.actions import get_actions
from src.explain import explain_customer, plot_waterfall
from src.predict import load_pipeline, predict_single, _apply_engineering, _drop_non_features
from src.roi import customer_value, expected_saving, format_roi_summary
from src.survival import churn_probability_windows, fit_kaplan_meier, plot_risk_windows

st.set_page_config(page_title="Customer Lookup — StayWise", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);color:#e2e8f0;}
[data-testid="stSidebar"]{background:rgba(15,23,42,0.95);}
.sw-card{background:rgba(30,41,59,0.7);border:1px solid rgba(14,165,233,0.15);border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
.badge-low{background:#064e3b;color:#6ee7b7;padding:6px 16px;border-radius:20px;font-size:1rem;font-weight:700;}
.badge-medium{background:#451a03;color:#fcd34d;padding:6px 16px;border-radius:20px;font-size:1rem;font-weight:700;}
.badge-high{background:#450a0a;color:#fca5a5;padding:6px 16px;border-radius:20px;font-size:1rem;font-weight:700;}
.badge-critical{background:#4c0519;color:#f9a8d4;padding:6px 16px;border-radius:20px;font-size:1rem;font-weight:700;}
</style>""", unsafe_allow_html=True)

st.markdown("## 🔍 Customer Lookup")
st.caption("Enter customer details to predict churn risk and get a recommended action")

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def _get_payload():
    try:
        return load_pipeline()
    except FileNotFoundError:
        return None

payload = _get_payload()
if payload is None:
    st.error("❌ No trained model found. Please go to **🧪 Model Lab** and train first.")
    st.stop()

meta = payload["metadata"]
threshold = meta.get("threshold", 0.5)

# ─── Load KM for survival (optional) ─────────────────────────────────────────
@st.cache_resource
def _get_km():
    try:
        from src.data import clean, load_raw
        raw = load_raw()
        df = clean(raw)
        return fit_kaplan_meier(df)
    except Exception:
        return None

kmf = _get_km()

# ─── Input form ───────────────────────────────────────────────────────────────
with st.expander("📝 Customer Details Form", expanded=True):
    st.markdown("Fill in the customer's current details:")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure, step=10.0)
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with c3:
        phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bkp = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

if predict_btn:
    record = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_svc,
        "MultipleLines": multi_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
        "DeviceProtection": device_prot, "TechSupport": tech_sup,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_mv,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }

    with st.spinner("Predicting …"):
        try:
            result = predict_single(record, payload=payload, threshold=threshold)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    prob = result["probability"]
    label = result["label"]
    risk = result["risk_category"]

    # ── Result header ──────────────────────────────────────────────────────────
    st.divider()
    col_gauge, col_detail = st.columns([1, 2])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Churn Probability", "font": {"size": 16, "family": "Inter"}},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.25 else "#10b981"},
                "bgcolor": "rgba(30,41,59,0.7)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(16,185,129,0.15)"},
                    {"range": [25, 60], "color": "rgba(245,158,11,0.15)"},
                    {"range": [60, 80], "color": "rgba(239,68,68,0.15)"},
                    {"range": [80, 100], "color": "rgba(239,68,68,0.3)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "value": threshold * 100},
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e2e8f0"),
            height=250, margin=dict(t=40, b=0, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        badge_cls = f"badge-{risk.lower()}"
        st.markdown(f"<div style='text-align:center'><span class='{badge_cls}'>{risk} Risk</span></div>",
                    unsafe_allow_html=True)
        st.caption(f"Threshold: {threshold:.0%} | Label: {'⚠️ Churn' if label else '✅ Retain'}")

    with col_detail:
        # ROI
        cval = customer_value(monthly, tenure)
        roi = expected_saving(prob, cval)
        st.markdown(f"""
        <div class="sw-card">
            <div style="font-size:1rem;font-weight:600;color:#0ea5e9;margin-bottom:0.8rem;">💰 ROI Analysis</div>
            <table style="width:100%;font-size:0.9rem;color:#cbd5e1;">
                <tr><td>Customer Value (est.)</td><td style="text-align:right;font-weight:600;">${cval:,.2f}</td></tr>
                <tr><td>Offer Cost</td><td style="text-align:right;">${roi['offer_cost']:.2f}</td></tr>
                <tr><td>Success Rate</td><td style="text-align:right;">{roi['success_rate']:.0%}</td></tr>
                <tr><td style="color:#0ea5e9;font-weight:600;">Expected Saving</td>
                    <td style="text-align:right;font-weight:700;color:{'#10b981' if roi['expected_saving']>0 else '#ef4444'};">
                        ${roi['expected_saving']:+,.2f}</td></tr>
            </table>
            <div style="margin-top:0.8rem;padding:0.6rem;background:rgba(14,165,233,0.1);border-radius:8px;font-size:0.85rem;">
                {"✅ <strong>Recommend action</strong>" if roi['recommend_action'] else "❌ <strong>No action</strong> — not cost-effective"}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Survival windows ───────────────────────────────────────────────────────
    windows = churn_probability_windows(kmf, tenure)
    st.plotly_chart(plot_risk_windows(windows, tenure), use_container_width=True)
    st.caption(windows.get("note", ""))

    # ── SHAP explanation ───────────────────────────────────────────────────────
    st.markdown("### 🧠 Why is this customer at risk?")
    record_df = pd.DataFrame([record])
    record_df = _drop_non_features(record_df)
    record_df = _apply_engineering(record_df)

    with st.spinner("Computing SHAP explanations …"):
        drivers = explain_customer(payload["pipeline"], record_df)

    if drivers:
        st.plotly_chart(plot_waterfall(drivers), use_container_width=True)
        st.markdown("**Key drivers:**")
        for d in drivers:
            icon = "🔴" if d["shap_value"] > 0 else "🟢"
            st.markdown(f"- {icon} **{d['reason']}** (SHAP: {d['shap_value']:+.3f})")

        # ── Actions ────────────────────────────────────────────────────────────
        st.markdown("### 🎯 Recommended Actions")
        actions = get_actions(drivers)
        for a in actions:
            with st.container():
                st.markdown(f"""
                <div class="sw-card">
                    <div style="font-weight:700;color:#0ea5e9;">🎁 {a['offer']}</div>
                    <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.4rem;">{a['rationale']}</div>
                    <div style="color:#64748b;font-size:0.8rem;margin-top:0.3rem;">💵 {a['cost_note']}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("SHAP explanation unavailable for this model configuration. Showing feature importance instead.")

    st.caption(
        "⚠️ SHAP values describe model behaviour, not causal proof. "
        "Actions are rule-based recommendations, not guaranteed outcomes."
    )
