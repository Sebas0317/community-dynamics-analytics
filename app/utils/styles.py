"""Editorial dark theme CSS and reusable layout helpers for Streamlit."""

from __future__ import annotations

import streamlit as st

PALETTE = {
    "primary": "#00D4AA",
    "secondary": "#38BDF8",
    "accent": "#F59E0B",
    "danger": "#F43F5E",
    "purple": "#A78BFA",
    "emerald": "#10B981",
    "slate": "#94A3B8",
    "bg": "#0B0F19",
    "card": "#131A2A",
    "border": "#1E293B",
    "text": "#F1F5F9",
}

CATEGORICAL_COLORS = [
    "#00D4AA",
    "#38BDF8",
    "#A78BFA",
    "#F59E0B",
    "#F43F5E",
    "#10B981",
    "#EC4899",
    "#60A5FA",
]


def setup_page(title: str, icon: str = "📊") -> None:
    st.set_page_config(
        page_title=f"{title} | Community Dynamics Analytics",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        .main .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }}
        h1, h2, h3 {{
            letter-spacing: -0.02em;
            font-weight: 700;
        }}
        .hero-banner {{
            background: linear-gradient(135deg, #131A2A 0%, #0F2333 60%, #0B2A2B 100%);
            border: 1px solid {PALETTE['border']};
            border-left: 4px solid {PALETTE['primary']};
            border-radius: 12px;
            padding: 1.4rem 1.7rem;
            margin-bottom: 1.4rem;
        }}
        .hero-tag {{
            display: inline-block;
            background: rgba(0, 212, 170, 0.14);
            color: {PALETTE['primary']};
            border: 1px solid rgba(0, 212, 170, 0.35);
            border-radius: 999px;
            padding: 0.2rem 0.75rem;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}
        .q-header {{
            background: #131A2A;
            border: 1px solid {PALETTE['border']};
            border-left: 4px solid {PALETTE['secondary']};
            border-radius: 10px;
            padding: 0.85rem 1.15rem;
            margin-top: 1.6rem;
            margin-bottom: 0.9rem;
        }}
        .q-badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.16);
            color: {PALETTE['secondary']};
            font-weight: 700;
            font-size: 0.78rem;
            padding: 0.15rem 0.55rem;
            border-radius: 6px;
            margin-right: 0.5rem;
        }}
        .insight-box {{
            background: rgba(0, 212, 170, 0.07);
            border: 1px solid rgba(0, 212, 170, 0.28);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-top: 0.6rem;
            margin-bottom: 1rem;
            font-size: 0.94rem;
            line-height: 1.5;
        }}
        .warn-box {{
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.30);
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-top: 0.5rem;
            font-size: 0.92rem;
        }}
        [data-testid="stMetric"] {{
            background: #131A2A;
            border: 1px solid {PALETTE['border']};
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }}
        [data-testid="stMetricValue"] {{
            color: {PALETTE['text']};
            font-weight: 700;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid {PALETTE['border']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(tag: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-tag">{tag}</div>
            <h1 style="margin: 0.15rem 0 0.45rem 0; font-size: 1.9rem;">{title}</h1>
            <p style="margin: 0; color: #CBD5E1; font-size: 0.98rem; line-height: 1.55;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_question_header(q_id: str, question: str, method: str) -> None:
    st.markdown(
        f"""
        <div class="q-header">
            <div><span class="q-badge">{q_id}</span><strong style="font-size: 1.08rem; color: #F8FAFC;">{question}</strong></div>
            <div style="color: #94A3B8; font-size: 0.83rem; margin-top: 0.25rem;">Metodología: {method}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight-box">
            <strong style="color: #00D4AA;">💡 {title}:</strong> {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer(date_range: dict | None = None) -> None:
    with st.sidebar:
        st.divider()
        st.caption("🔒 **Privacidad Zero-PII (Ley 1581/2012)**")
        st.caption("Identidades anonimizadas con `HMAC-SHA256` efímero. Cero texto crudo ni fotos publicadas.")
        if date_range:
            st.caption(f"📅 Corte: `{date_range.get('min_dt', '')[:10]}` → `{date_range.get('max_dt', '')[:10]}`")
        st.caption("Autor: **Kevin Sebastián (`Sebas0317`)**")
