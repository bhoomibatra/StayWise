"""
src/ui_theme.py
---------------
Centralized UI/UX design system and theme components for StayWise.
Implements modern Cyber-SaaS dark aesthetics, glassmorphism, glowing badges,
and standardized Plotly visualization themes.
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

# ─── Color Tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "bg_dark": "#080c14",
    "card_bg": "rgba(18, 26, 43, 0.75)",
    "card_border": "rgba(0, 240, 255, 0.18)",
    "cyan": "#00f0ff",
    "violet": "#8b5cf6",
    "emerald": "#10b981",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
}

def inject_custom_css() -> None:
    """Inject unified modern styling across all pages."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Main application background with subtle radial glow */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
                    linear-gradient(135deg, #080c14 0%, #0e1626 50%, #080c14 100%) !important;
        color: #f1f5f9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 26, 0.95) !important;
        border-right: 1px solid rgba(0, 240, 255, 0.15) !important;
        backdrop-filter: blur(12px);
    }
    
    /* Modern Glassmorphic Cards */
    .sw-glass-card {
        background: rgba(18, 26, 43, 0.7) !important;
        border: 1px solid rgba(0, 240, 255, 0.15) !important;
        border-radius: 14px !important;
        padding: 1.4rem !important;
        margin-bottom: 1.2rem !important;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .sw-glass-card:hover {
        border-color: rgba(0, 240, 255, 0.35) !important;
        box-shadow: 0 10px 40px -10px rgba(0, 240, 255, 0.15);
        transform: translateY(-2px);
    }

    /* KPI Tiles */
    .sw-kpi-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .sw-kpi-num {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f0ff 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .sw-kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .sw-kpi-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* Risk Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-high {
        background: rgba(244, 63, 94, 0.18);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }
    .badge-critical {
        background: rgba(225, 29, 72, 0.25);
        color: #fda4af;
        border: 1px solid rgba(225, 29, 72, 0.6);
        box-shadow: 0 0 15px rgba(225, 29, 72, 0.4);
        animation: pulse 1.8s infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.85; }
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 240, 255, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 240, 255, 0.4) !important;
        border-color: #00f0ff !important;
    }

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(14, 22, 38, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stTabs"] button {
        border-radius: 8px !important;
        padding: 8px 16px !important;
        color: #94a3b8 !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        background: rgba(0, 240, 255, 0.12) !important;
        color: #00f0ff !important;
        font-weight: 600 !important;
    }

    /* Inputs & Selectboxes */
    div[data-baseweb="select"] > div,
    input[type="text"], input[type="number"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    div[data-baseweb="select"] > div:hover,
    input[type="text"]:focus, input[type="number"]:focus {
        border-color: #00f0ff !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def get_plotly_layout() -> dict:
    """Standardized dark layout parameters for Plotly figures."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#cbd5e1"),
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
    )


def render_page_header(title: str, subtitle: str, tag: str | None = None) -> None:
    """Render a modern gradient page header."""
    badge_html = f'<span style="background: rgba(0, 240, 255, 0.12); color: #00f0ff; border: 1px solid rgba(0, 240, 255, 0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.6rem; display: inline-block;">{tag}</span>' if tag else ""
    st.markdown(f"""
    <div style="margin-bottom: 1.6rem; padding-bottom: 0.8rem; border-bottom: 1px solid rgba(255, 255, 255, 0.07);">
        {badge_html}
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #00f0ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
            {title}
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)
