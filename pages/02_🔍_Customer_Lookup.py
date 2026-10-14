"""
pages/02_🔍_Customer_Lookup.py — StayWise Customer Lookup
Single-customer churn prediction with 1-click presets, SHAP attributions,
survival analysis, and personalized retention ROI.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.actions import get_actions
from src.explain import explain_customer, plot_waterfall
from src.predict import load_pipeline, predict_single, _apply_engineering, _drop_non_features
from src.roi import customer_value, expected_saving
from src.survival import churn_probability_windows, fit_kaplan_meier, plot_risk_windows
from src.ui_theme import inject_custom_css, get_plotly_layout, render_page_header

st.set_page_config(page_title="Customer Lookup — StayWise", page_icon="🔍", layout="wide")

# ─── Global Styles ────────────────────────────────────────────────────────────
inject_custom_css()
render_page_header("Customer Risk Scoring & SHAP Explainer", "Evaluate individual churn probability, uncover underlying drivers, and determine optimal retention interventions", "1-to-1 Intelligence")

# ─── Load Model & Survival Engine ─────────────────────────────────────────────
@st.cache_resource
def _get_payload():
    try:
        return load_pipeline()
    except FileNotFoundError:
        return None

payload = _get_payload()
if payload is None:
    st.error("❌ No trained model found. Please train models in **🧪 Model Lab** first.")
    st.stop()

meta = payload["metadata"]
threshold = meta.get("threshold", 0.06)

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

# ─── 1-Click Customer Presets ─────────────────────────────────────────────────
st.markdown("#### ⚡ 1-Click Customer Archetypes")
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

PRESETS = {
    "high_risk": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 94.50, "TotalCharges": 189.0,
    },
    "loyal_family": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 64, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 62.0, "TotalCharges": 3968.0,
    },
    "senior_streamer": {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 14, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 101.50, "TotalCharges": 1421.0,
    }
}

if "form_data" not in st.session_state:
    st.session_state["form_data"] = PRESETS["high_risk"]

with p_col1:
    if st.button("🔥 High-Risk New User", use_container_width=True):
        st.session_state["form_data"] = PRESETS["high_risk"]
        st.rerun()

with p_col2:
    if st.button("🛡️ Loyal 2-Year Family", use_container_width=True):
        st.session_state["form_data"] = PRESETS["loyal_family"]
        st.rerun()

with p_col3:
    if st.button("⚠️ At-Risk Senior Streamer", use_container_width=True):
        st.session_state["form_data"] = PRESETS["senior_streamer"]
        st.rerun()

with p_col4:
    if st.button("🔄 Reset to Default", use_container_width=True):
        st.session_state["form_data"] = PRESETS["high_risk"]
        st.rerun()

current = st.session_state["form_data"]

# ─── Input Form ───────────────────────────────────────────────────────────────
with st.expander("📝 Customer Profile Parameters", expanded=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if current["gender"] == "Male" else 1)
        senior = st.selectbox("Senior Citizen", [0, 1], index=current["SeniorCitizen"], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"], index=0 if current["Partner"] == "Yes" else 1)
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if current["Dependents"] == "Yes" else 1)
        tenure = st.slider("Tenure (months)", 0, 72, int(current["tenure"]))

    with c2:
        contracts = ["Month-to-month", "One year", "Two year"]
        contract = st.selectbox("Contract Type", contracts, index=contracts.index(current["Contract"]))
        monthly = st.number_input("Monthly Charges ($)", 18.0, 130.0, float(current["MonthlyCharges"]), step=0.5)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(current["TotalCharges"]), step=10.0)
        internets = ["Fiber optic", "DSL", "No"]
        internet = st.selectbox("Internet Service", internets, index=internets.index(current["InternetService"]))
        payments = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        payment = st.selectbox("Payment Method", payments, index=payments.index(current["PaymentMethod"]))

    with c3:
        phone_svc = st.selectbox("Phone Service", ["Yes", "No"], index=0 if current["PhoneService"] == "Yes" else 1)
        multi_opts = ["Yes", "No", "No phone service"]
        multi_lines = st.selectbox("Multiple Lines", multi_opts, index=multi_opts.index(current["MultipleLines"]))
        opts = ["Yes", "No", "No internet service"]
        online_sec = st.selectbox("Online Security", opts, index=opts.index(current["OnlineSecurity"]))
        online_bkp = st.selectbox("Online Backup", opts, index=opts.index(current["OnlineBackup"]))
        device_prot = st.selectbox("Device Protection", opts, index=opts.index(current["DeviceProtection"]))
        tech_sup = st.selectbox("Tech Support", opts, index=opts.index(current["TechSupport"]))
        streaming_tv = st.selectbox("Streaming TV", opts, index=opts.index(current["StreamingTV"]))
        streaming_mv = st.selectbox("Streaming Movies", opts, index=opts.index(current["StreamingMovies"]))
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if current["PaperlessBilling"] == "Yes" else 1)

# ─── Inference Execution ──────────────────────────────────────────────────────
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

result = predict_single(record, payload=payload, threshold=threshold)
prob = result["probability"]
label = result["label"]
risk = result["risk_category"]

# ─── Results Dashboard ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Predictive Assessment")

col_gauge, col_roi = st.columns([1, 1])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Calibrated Churn Probability", "font": {"size": 17, "family": "Plus Jakarta Sans", "color": "#f1f5f9"}},
        number={"suffix": "%", "font": {"size": 42, "color": "#00f0ff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": "#f43f5e" if prob > 0.6 else "#f59e0b" if prob > 0.25 else "#10b981", "thickness": 0.28},
            "bgcolor": "rgba(18, 26, 43, 0.7)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 25], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [25, 60], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [60, 80], "color": "rgba(244, 63, 94, 0.2)"},
                {"range": [80, 100], "color": "rgba(225, 29, 72, 0.35)"},
            ],
            "threshold": {"line": {"color": "#00f0ff", "width": 3}, "value": threshold * 100},
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#e2e8f0"),
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    badge_map = {
        "Low": "badge-low", "Medium": "badge-medium",
        "High": "badge-high", "Critical": "badge-critical"
    }
    st.markdown(f"""
    <div style="text-align:center; margin-top:-10px;">
        <span class="badge-pill {badge_map.get(risk, 'badge-medium')}">{risk} Risk Classification</span>
        <div style="font-size:0.8rem; color:#94a3b8; margin-top:8px;">
            Decision Cutoff: <strong>{threshold:.1%}</strong> | Predicted State: <strong style="color:{'#f43f5e' if label else '#34d399'};">{'Defection Alert' if label else 'Customer Safe'}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_roi:
    cval = customer_value(monthly, tenure)
    roi = expected_saving(prob, cval)
    st.markdown(f"""
    <div class="sw-glass-card" style="height:100%;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
            <span style="font-weight:700; color:#00f0ff; text-transform:uppercase; font-size:0.85rem; letter-spacing:0.05em;">Financial Utility & Retention ROI</span>
            <span style="font-size:0.8rem; color:#94a3b8;">Offer Cost: ${roi['offer_cost']:.0f}</span>
        </div>
        <table style="width:100%; font-size:0.9rem; color:#cbd5e1; border-collapse:collapse;">
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:8px 0;">Estimated Lifetime Value (CLV)</td>
                <td style="text-align:right; font-weight:700; color:#ffffff;">${cval:,.2f}</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:8px 0;">Intervention Success Rate</td>
                <td style="text-align:right; font-weight:700; color:#ffffff;">{roi['success_rate']:.0%}</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                <td style="padding:8px 0;">Expected Value Saved</td>
                <td style="text-align:right; font-weight:700; color:#ffffff;">${prob * cval * roi['success_rate']:,.2f}</td>
            </tr>
            <tr>
                <td style="padding:10px 0; font-weight:700; color:#00f0ff;">Net Expected Savings</td>
                <td style="text-align:right; font-size:1.1rem; font-weight:800; color:{'#34d399' if roi['expected_saving'] > 0 else '#f43f5e'};">
                    ${roi['expected_saving']:+,.2f}
                </td>
            </tr>
        </table>
        <div style="margin-top:1rem; padding:0.7rem 1rem; border-radius:8px; background:{'rgba(16, 185, 129, 0.15)' if roi['recommend_action'] else 'rgba(244, 63, 94, 0.15)'}; border:1px solid {'rgba(16, 185, 129, 0.3)' if roi['recommend_action'] else 'rgba(244, 63, 94, 0.3)'};">
            <span style="font-weight:700; color:{'#34d399' if roi['recommend_action'] else '#f43f5e'};">
                {'✅ Recommended: Dispatch Retention Offer' if roi['recommend_action'] else '⏸️ Hold Action: Low financial payoff'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── SHAP Drivers & Actions ───────────────────────────────────────────────────
st.markdown("### 🧠 Local Explainability (SHAP Attributions)")

record_df = pd.DataFrame([record])
record_df = _drop_non_features(record_df)
record_df = _apply_engineering(record_df)

with st.spinner("Calculating SHAP attributions …"):
    drivers = explain_customer(payload["pipeline"], record_df)

if drivers:
    col_wfall, col_actions = st.columns([1, 1])
    with col_wfall:
        fig_waterfall = plot_waterfall(drivers)
        fig_waterfall.update_layout(**get_plotly_layout())
        st.plotly_chart(fig_waterfall, use_container_width=True)
    with col_actions:
        st.markdown("#### 🎯 Prescribed Interventions")
        actions = get_actions(drivers)
        for act in actions:
            st.markdown(f"""
            <div class="sw-glass-card" style="margin-bottom:0.8rem; padding:1rem;">
                <div style="font-weight:700; color:#00f0ff; font-size:0.95rem;">🎁 {act['offer']}</div>
                <div style="color:#cbd5e1; font-size:0.85rem; margin-top:4px;">{act['rationale']}</div>
                <div style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">Cost: {act['cost_note']}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("SHAP attribution engine loaded fallback mode.")

# ─── Survival Window Analysis ─────────────────────────────────────────────────
if kmf is not None:
    st.markdown("### ⏳ Hazard Over Time (Survival Probability)")
    windows = churn_probability_windows(kmf, tenure)
    fig_surv = plot_risk_windows(windows, tenure)
    fig_surv.update_layout(**get_plotly_layout())
    st.plotly_chart(fig_surv, use_container_width=True)
