"""
pages/03_🎛️_What_If.py — StayWise What-If Simulator
Change customer attributes with sliders and see real-time churn risk delta.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import load_pipeline, predict_single

st.set_page_config(page_title="What-If — StayWise", page_icon="🎛️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);color:#e2e8f0;}
[data-testid="stSidebar"]{background:rgba(15,23,42,0.95);}
.sw-card{background:rgba(30,41,59,0.7);border:1px solid rgba(14,165,233,0.15);border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
</style>""", unsafe_allow_html=True)

st.markdown("## 🎛️ What-If Simulator")
st.caption("Adjust customer attributes and see how churn probability changes in real time")

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

st.markdown("### 🧍 Baseline Customer")
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12, key="base_tenure")
    monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, key="base_monthly")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="base_contract")

with col2:
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="base_internet")
    tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="base_tech")
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="base_sec")
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], key="base_payment")

def _build_record(tenure_, monthly_, contract_, internet_, tech_sup_, online_sec_, payment_) -> dict:
    return {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": tenure_, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": internet_, "OnlineSecurity": online_sec_,
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": tech_sup_,
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": contract_,
        "PaperlessBilling": "Yes", "PaymentMethod": payment_,
        "MonthlyCharges": monthly_, "TotalCharges": monthly_ * max(tenure_, 1),
    }

baseline_record = _build_record(tenure, monthly, contract, internet, tech_sup, online_sec, payment)
baseline_result = predict_single(baseline_record, payload=payload, threshold=threshold)
baseline_prob = baseline_result["probability"]

st.divider()
st.markdown("### 🔄 Simulate Changes")

c1, c2 = st.columns(2)
with c1:
    new_contract = st.selectbox("New Contract", ["Month-to-month", "One year", "Two year"],
                                index=["Month-to-month", "One year", "Two year"].index(contract),
                                key="new_contract")
    new_monthly = st.slider("New Monthly Charges ($)", 18.0, 120.0, monthly, key="new_monthly")
    new_tenure = st.slider("New Tenure (months)", 0, 72, tenure, key="new_tenure")

with c2:
    new_tech = st.selectbox("New Tech Support", ["No", "Yes", "No internet service"],
                             index=["No", "Yes", "No internet service"].index(tech_sup),
                             key="new_tech")
    new_sec = st.selectbox("New Online Security", ["No", "Yes", "No internet service"],
                            index=["No", "Yes", "No internet service"].index(online_sec),
                            key="new_sec")
    new_payment = st.selectbox("New Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(payment),
    key="new_payment")

new_record = _build_record(new_tenure, new_monthly, new_contract, internet, new_tech, new_sec, new_payment)
new_result = predict_single(new_record, payload=payload, threshold=threshold)
new_prob = new_result["probability"]
delta = new_prob - baseline_prob

# ── Side-by-side gauges ────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Results")

ga, gb, gc = st.columns(3)

def _gauge(title, prob, threshold_):
    color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.25 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": title, "font": {"size": 14, "family": "Inter"}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "rgba(30,41,59,0.7)",
            "threshold": {"line": {"color": "#fff", "width": 2}, "value": threshold_ * 100},
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                      height=200, margin=dict(t=40, b=0, l=10, r=10))
    return fig

with ga:
    st.plotly_chart(_gauge("Baseline Risk", baseline_prob, threshold), use_container_width=True)
    st.caption(f"Risk: {baseline_result['risk_category']}")
with gb:
    st.plotly_chart(_gauge("New Risk", new_prob, threshold), use_container_width=True)
    st.caption(f"Risk: {new_result['risk_category']}")
with gc:
    direction = "🔴 Increased" if delta > 0.01 else "🟢 Reduced" if delta < -0.01 else "⚪ No Change"
    st.markdown(f"""
    <div class="sw-card" style="text-align:center;margin-top:20px;">
        <div style="font-size:2.5rem;font-weight:700;color:{'#ef4444' if delta>0.01 else '#10b981' if delta<-0.01 else '#94a3b8'};">
            {delta:+.1%}
        </div>
        <div style="color:#94a3b8;margin-top:0.5rem;">Risk Delta</div>
        <div style="margin-top:0.8rem;">{direction}</div>
    </div>""", unsafe_allow_html=True)

if abs(delta) > 0.01:
    if delta < 0:
        st.success(f"✅ This change **reduces churn risk by {abs(delta):.1%}**. Consider implementing it.")
    else:
        st.warning(f"⚠️ This change **increases churn risk by {abs(delta):.1%}**.")
else:
    st.info("ℹ️ This change has minimal impact on churn probability.")
