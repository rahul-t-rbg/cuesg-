from __future__ import annotations

import html
import json
import os
from io import BytesIO, StringIO
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from greenlens_ai_core import (
    AdvancedAI,
    DeterministicMath,
    GREENLENS_SYSTEM_METADATA,
    RegulatoryIntelligence,
    SessionSustainabilityTracker,
    generate_synthetic_demo_data,
    load_master_dataset,
)

st.set_page_config(
    page_title="CUESG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MAX_CHAT_HISTORY = 80


def inject_css(control_room: bool) -> None:
    chat_display = "none" if control_room else "flex"
    # Inject font link tags separately — more reliable than @import alone in Streamlit
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )

    st.markdown(
f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
    --font-ui: 'Manrope', 'Segoe UI', sans-serif;
    --font-display: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
    --font-mono: 'IBM Plex Mono', 'SF Mono', 'Consolas', 'Courier New', monospace;
}}

/* =============================================
   CUESG — CONTROL ROOM INTERFACE
   ============================================= */

/* 1. Reset & Base */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}

[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

.stApp {{
    background:
        radial-gradient(ellipse 80% 40% at 10% 0%, rgba(0, 255, 127, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(88, 166, 255, 0.06) 0%, transparent 60%),
        #060a0f;
    font-family: var(--font-ui);
    color: #c9d1d9;
    background-attachment: fixed;
}}

/* Force all Streamlit text elements to use our font stack */
.stApp p, .stApp li, .stApp span, .stApp label,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp .stMarkdown p {{ 
    font-family: var(--font-ui) !important;
    font-size: 0.82rem !important;
    line-height: 1.75 !important;
    color: #8b949e;
}}

/* Headings use the display font */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 {{
    font-family: var(--font-display) !important;
    color: #f0f6fc !important;
    letter-spacing: -0.02em;
}}

/* Bold in markdown */
.stApp strong, .stApp b {{
    color: #c9d1d9 !important;
    font-weight: 600;
}}

/* Inline code */
.stApp code {{
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    background: rgba(88, 166, 255, 0.08) !important;
    border: 1px solid rgba(88, 166, 255, 0.15) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    color: #79c0ff !important;
}}

.block-container {{
    max-width: 100% !important;
    padding: 1.5rem 2.5rem 7rem 2.5rem !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(88,166,255,0.25); border-radius: 8px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(88,166,255,0.5); }}

/* =============================================
   2. TOPBAR SYSTEM HEADER
   ============================================= */
.sys-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0 24px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 28px;
}}

.sys-wordmark {{
    font-family: var(--font-display);
    font-size: 0.85rem;
    font-weight: 700;
    color: #00ff7f;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}}

.sys-status {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-ui);
    font-size: 0.7rem;
    color: #4d9375;
    letter-spacing: 0.1em;
}}

.sys-status::before {{
    content: "";
    display: block;
    width: 6px;
    height: 6px;
    background: #00ff7f;
    border-radius: 50%;
    box-shadow: 0 0 8px #00ff7f;
    animation: pulse 2s ease-in-out infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 8px #00ff7f; }}
    50% {{ opacity: 0.4; box-shadow: 0 0 4px #00ff7f; }}
}}

/* =============================================
   3. HERO — CLEAN CENTERED, CONTROLLED HEIGHT
   ============================================= */
.hero-container {{
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
    animation: fadeUp 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}}

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.hero-title {{
    font-family: var(--font-display);
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #ffffff 20%, #00ff7f 60%, #58a6ff 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}}

.hero-copy {{
    font-family: var(--font-ui);
    color: #4d5a6a;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    max-width: 56ch;
    margin: 0 auto;
    line-height: 1.8;
}}

/* =============================================
   4. HINT CARDS — 2-COL, NO OVERFLOW
   ============================================= */
.hint-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1.2rem 0 1.4rem 0;
}}

.hint-card {{
    background: rgba(13, 17, 23, 0.7);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-left: 3px solid rgba(0, 255, 127, 0.25);
    padding: 18px 22px;
    border-radius: 10px;
    transition: border-left-color 0.2s ease, background 0.2s ease;
    cursor: pointer;
}}

.hint-card:hover {{
    border-left-color: #00ff7f;
    background: rgba(0, 255, 127, 0.03);
}}

.hint-title {{
    font-family: var(--font-display);
    color: #00ff7f;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-weight: 700;
    margin-bottom: 8px;
}}

.hint-copy {{
    color: #8b949e;
    font-size: 0.85rem;
    line-height: 1.5;
    font-family: var(--font-ui);
}}

/* =============================================
   5. UPLOAD PANEL — UNDER HERO
   ============================================= */
.upload-panel {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 0 0 1.6rem 0;
}}

.upload-block {{
    background: rgba(13, 17, 23, 0.7);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-radius: 10px;
    padding: 18px 22px;
}}

/* =============================================
   6. CHAT SHELL — CONTAINED, NO OVERFLOW
   ============================================= */
.chat-shell {{
    display: {chat_display};
    flex-direction: column;
    gap: 12px;
    min-height: 120px;
    max-height: 480px;
    overflow-y: auto;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(48, 54, 61, 0.5);
    background: rgba(6, 10, 15, 0.7);
    margin-bottom: 20px;
}}

.message-card {{
    border-radius: 10px;
    padding: 14px 18px;
    max-width: 78%;
    animation: popIn 0.25s ease forwards;
}}

@keyframes popIn {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.message-card.assistant {{
    background: rgba(13, 17, 23, 0.9);
    border: 1px solid rgba(48, 54, 61, 0.6);
    border-left: 2px solid rgba(0, 255, 127, 0.4);
    align-self: flex-start;
}}

.message-card.user {{
    background: rgba(10, 22, 17, 0.9);
    border: 1px solid rgba(23, 83, 49, 0.6);
    border-right: 2px solid rgba(88, 166, 255, 0.4);
    align-self: flex-end;
}}

.role {{
    font-family: var(--font-display);
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 8px;
    color: #00ff7f;
}}

.message-card.user .role {{
    color: #58a6ff;
    text-align: right;
}}

.body {{
    color: #c9d1d9;
    line-height: 1.65;
    white-space: pre-wrap;
    font-size: 0.9rem;
    font-family: var(--font-ui);
}}

/* =============================================
   7. CHAT INPUT — DOCKED, CLEAN
   ============================================= */
div[data-testid="stChatInput"] {{
    position: fixed;
    bottom: 1.5rem;
    left: 50%;
    transform: translateX(-50%);
    width: min(720px, 90vw);
    z-index: 999;
    background: rgba(13, 17, 23, 0.95);
    border: 1px solid rgba(48, 54, 61, 0.8);
    border-radius: 12px;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(0, 255, 127, 0.05);
}}

div[data-testid="stChatInput"] textarea {{
    font-family: var(--font-ui) !important;
    font-size: 0.85rem !important;
    color: #c9d1d9 !important;
    background: transparent !important;
}}

/* =============================================
   7. GLASS PANEL — SUBTLE, CLEAN
   ============================================= */
.glass-panel {{
    background: rgba(13, 17, 23, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color 0.2s ease;
}}

.glass-panel:hover {{
    border-color: rgba(88, 166, 255, 0.15);
}}

/* =============================================
   8. METRIC CARDS — TIGHT & CLEAN
   ============================================= */
.metric-card {{
    background: rgba(13, 17, 23, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-top: 1px solid rgba(0, 255, 127, 0.15);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-top-color 0.2s ease;
}}

.metric-card:hover {{
    border-top-color: rgba(0, 255, 127, 0.4);
}}

.metric-label {{
    font-family: var(--font-display);
    color: #4d5a6a;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 8px;
}}

.metric-value {{
    font-family: var(--font-display);
    color: #f0f6fc;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}}

/* =============================================
   9. TABS — REFINED
   ============================================= */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    margin-bottom: 20px;
    background: rgba(13, 17, 23, 0.6);
    border-radius: 8px;
    border: 1px solid rgba(48, 54, 61, 0.4);
    padding: 4px;
}}

.stTabs [data-baseweb="tab"] {{
    font-family: var(--font-ui);
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    padding: 8px 14px;
    border-radius: 6px;
    font-weight: 500;
    color: #4d5a6a;
    border: none;
    transition: all 0.15s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: rgba(48, 54, 61, 0.4);
    color: #8b949e;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: rgba(0, 255, 127, 0.08);
    color: #00ff7f;
    box-shadow: inset 0 0 0 1px rgba(0, 255, 127, 0.2);
}}

/* =============================================
   11. DATA TABLES & CHARTS
   ============================================= */
div[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(48, 54, 61, 0.4);
    margin-bottom: 16px;
}}

div[data-testid="stPlotlyChart"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(48, 54, 61, 0.4);
    background: rgba(6, 10, 15, 0.4);
    margin-bottom: 16px;
}}

/* =============================================
   12. STREAMLIT BUTTONS — CONSISTENT
   ============================================= */
.stButton > button {{
    font-family: var(--font-ui) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(48, 54, 61, 0.6) !important;
    background: rgba(13, 17, 23, 0.8) !important;
    color: #8b949e !important;
    padding: 8px 14px !important;
    transition: all 0.15s ease !important;
    text-transform: uppercase !important;
}}

.stButton > button:hover {{
    border-color: rgba(0, 255, 127, 0.4) !important;
    color: #00ff7f !important;
    background: rgba(0, 255, 127, 0.05) !important;
}}

/* =============================================
   13. FILE UPLOADER — CLEAN
   ============================================= */
[data-testid="stFileUploader"] {{
    border: 1px dashed rgba(48, 54, 61, 0.6);
    border-radius: 10px;
    padding: 12px;
    background: rgba(6, 10, 15, 0.4);
    margin-bottom: 12px;
}}

[data-testid="stFileUploader"] label {{
    font-family: var(--font-ui) !important;
    font-size: 0.72rem !important;
    color: #4d5a6a !important;
    letter-spacing: 0.05em;
}}

/* =============================================
   14. SLIDERS
   ============================================= */
[data-testid="stSlider"] label {{
    font-family: var(--font-ui) !important;
    font-size: 0.72rem !important;
    color: #4d5a6a !important;
    letter-spacing: 0.05em;
}}

/* =============================================
   15. SELECTBOX
   ============================================= */
[data-testid="stSelectbox"] label {{
    font-family: var(--font-ui) !important;
    font-size: 0.72rem !important;
    color: #4d5a6a !important;
    letter-spacing: 0.05em;
}}

/* =============================================
   16. SECTION DIVIDERS
   ============================================= */
hr {{
    border: none;
    border-top: 1px solid rgba(48, 54, 61, 0.3);
    margin: 24px 0;
}}

/* =============================================
   17. CONTROL ROOM HEADER
   ============================================= */
.ctrl-header {{
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 16px;
    padding: 20px;
    background: rgba(13, 17, 23, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-radius: 12px;
    margin-bottom: 20px;
}}

.ctrl-cell {{
    padding: 4px 0;
}}

.ctrl-label {{
    font-family: var(--font-display);
    font-size: 0.58rem;
    color: #4d5a6a;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 6px;
}}

.ctrl-value {{
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}}

.ctrl-value.green {{ color: #00ff7f; text-shadow: 0 0 20px rgba(0,255,127,0.15); }}
.ctrl-value.blue {{ color: #58a6ff; text-shadow: 0 0 20px rgba(88,166,255,0.15); }}
.ctrl-value.white {{ color: #f0f6fc; }}

.ctrl-sub {{
    font-family: var(--font-ui);
    font-size: 0.7rem;
    color: #4d5a6a;
    margin-top: 4px;
    line-height: 1.6;
}}

/* =============================================
   18. ASSET NODE CARDS
   ============================================= */
.asset-card {{
    background: rgba(13, 17, 23, 0.8);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s ease;
}}

.asset-card:hover {{
    border-color: rgba(0, 255, 127, 0.25);
}}

.asset-name {{
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 10px;
}}

.asset-meta {{
    font-family: var(--font-ui);
    font-size: 0.72rem;
    color: #4d5a6a;
    line-height: 1.8;
}}

.asset-meta b {{
    color: #8b949e;
    font-weight: 500;
}}

/* =============================================
   19. DOWNLOAD BUTTONS (SPECIAL)
   ============================================= */
[data-testid="stDownloadButton"] > button {{
    font-family: var(--font-ui) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0, 255, 127, 0.25) !important;
    background: rgba(0, 255, 127, 0.05) !important;
    color: #00ff7f !important;
    text-transform: uppercase !important;
}}

[data-testid="stDownloadButton"] > button:hover {{
    background: rgba(0, 255, 127, 0.1) !important;
    border-color: rgba(0, 255, 127, 0.5) !important;
}}

/* =============================================
   20. JSON VIEWER
   ============================================= */
[data-testid="stJson"] {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    border: 1px solid rgba(48, 54, 61, 0.4);
    border-radius: 10px;
    background: rgba(6, 10, 15, 0.4);
}}

/* =============================================
   21. CODE BLOCKS
   ============================================= */
.stCode {{
    font-family: var(--font-mono);
    font-size: 0.75rem;
    border-radius: 10px;
    border: 1px solid rgba(48, 54, 61, 0.4);
}}

/* =============================================
   22. INFO / WARNING BOXES
   ============================================= */
[data-testid="stInfo"] {{
    font-family: var(--font-ui);
    font-size: 0.8rem;
    border-radius: 8px;
    border-left: 3px solid #58a6ff;
    background: rgba(88, 166, 255, 0.05);
}}

/* =============================================
   23. SPACING UTILITIES
   ============================================= */
.spacer-sm {{ height: 12px; }}
.spacer-md {{ height: 24px; }}
.spacer-lg {{ height: 40px; }}

/* =============================================
   24. EXPANDER — WHAT-IF CONTROLS
   ============================================= */
[data-testid="stExpander"] {{
    background: rgba(13, 17, 23, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.4);
    border-radius: 10px;
    margin-bottom: 16px;
}}

[data-testid="stExpander"] summary {{
    font-family: var(--font-ui) !important;
    font-size: 0.75rem !important;
    color: #4d5a6a !important;
    letter-spacing: 0.08em;
    padding: 12px 16px;
}}

[data-testid="stExpander"] summary:hover {{
    color: #8b949e !important;
}}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
    padding: 0 16px 16px 16px;
}}

/* =============================================
   24. ACTION BUTTON ROW — TOP OF PAGE
   ============================================= */
.action-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(48, 54, 61, 0.3);
}}

/* =============================================
   25. COMMAND SNAPSHOT HEADER
   ============================================= */
.cmd-snapshot {{
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 20px;
    padding: 20px;
    background: rgba(13, 17, 23, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-radius: 12px;
    margin-bottom: 16px;
}}

.cmd-title {{
    font-family: var(--font-display);
    font-size: 1.3rem;
    font-weight: 800;
    color: #f0f6fc;
    margin-bottom: 8px;
}}

.cmd-desc {{
    font-family: var(--font-ui);
    font-size: 0.75rem;
    color: #4d5a6a;
    line-height: 1.7;
}}

.cmd-tag {{
    font-family: var(--font-display);
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #00ff7f;
    text-transform: uppercase;
    margin-bottom: 10px;
}}

.cmd-stat-box {{
    background: rgba(6, 10, 15, 0.6);
    border: 1px solid rgba(48, 54, 61, 0.5);
    border-radius: 10px;
    padding: 18px;
}}

.cmd-stat-label {{
    font-family: var(--font-display);
    font-size: 0.6rem;
    color: #4d5a6a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 700;
}}

.cmd-stat-value {{
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 6px 0 10px 0;
}}

.cmd-stat-rows {{
    font-family: var(--font-ui);
    font-size: 0.72rem;
    color: #4d5a6a;
    line-height: 2;
}}

.cmd-stat-rows b {{ color: #8b949e; font-weight: 500; }}
.cmd-stat-rows b.green {{ color: #00ff7f; }}

</style>""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def boot():
    math_engine = DeterministicMath()
    ai_engine = AdvancedAI(math_engine)
    reg_engine = RegulatoryIntelligence(math_engine, ai_engine)
    carbon_tracker = SessionSustainabilityTracker()

    master = load_master_dataset()

    if not master.empty:
        master_calc, _ = math_engine.calculate(master)
        ai_engine.train_models(master_calc)

    return math_engine, ai_engine, reg_engine, carbon_tracker


math_engine, ai_engine, reg_engine, carbon_tracker = boot()


def ensure_state() -> None:
    defaults = {
        "messages": [{"role": "assistant", "content": "[OMNI-SYSTEM] Upload a dataset or load the master file. This chat is the primary command interface."}],
        "active_package": None,
        "active_signature": None,
        "pending_bill": None,
        "session_carbon_g": 0.0,
        "rolling_context": "",
        "control_room": False,
        "queued_prompt": "",
        "dossier_md": "",
        "dossier_pdf": b"",
        "cached_forecast_df": None,
        "_trigger_pdf_build": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ensure_state()
inject_css(st.session_state.control_room)


def fig_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color="#8b949e")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#8b949e", size=11),
        margin=dict(l=12, r=12, t=44, b=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )


def render_plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


def queue_prompt(prompt_text: str) -> None:
    st.session_state.queued_prompt = prompt_text


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded ESG dataset (CSV / Excel / PDF with tabular data).
    PDFs are treated as ESG data tables first; if that fails the user should
    use the separate 'Upload utility bill' uploader instead.
    """
    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
    payload = uploaded_file.getvalue()
    if ext == "csv":
        # Try common encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(BytesIO(payload), encoding=enc)
            except Exception:
                continue
        return pd.read_csv(BytesIO(payload), encoding="latin-1", errors="replace")
    if ext in {"xlsx", "xls"}:
        return pd.read_excel(BytesIO(payload))
    if ext == "pdf":
        # Try pdfplumber for table extraction first
        try:
            import pdfplumber, io as _io
            with pdfplumber.open(_io.BytesIO(payload)) as pdf:
                dfs = []
                for page in pdf.pages[:8]:
                    tables = page.extract_tables()
                    for tbl in tables:
                        if tbl and len(tbl) > 1:
                            try:
                                df_t = pd.DataFrame(tbl[1:], columns=tbl[0])
                                dfs.append(df_t)
                            except Exception:
                                pass
                if dfs:
                    return pd.concat(dfs, ignore_index=True)
        except Exception:
            pass
        # Fallback: raw text → try CSV parse
        try:
            decoded = payload.decode("latin-1", errors="ignore")
            lines   = [l.strip() for l in decoded.splitlines() if l.strip()]
            clean   = "\n".join(lines)
            try:
                return pd.read_csv(StringIO(clean))
            except Exception:
                return pd.read_csv(StringIO(clean), sep=r"\s{2,}", engine="python")
        except Exception:
            pass
        raise ValueError("Could not parse tabular data from PDF. If this is a utility bill, use the 'Upload utility bill' uploader below.")
    raise ValueError(f"Unsupported dataset format: .{ext}")



def activate_portfolio(raw_df: pd.DataFrame, source_name: str) -> None:
    calculated_df, detected_columns = math_engine.calculate(raw_df)
    ai_engine.train_models(calculated_df)
    package = ai_engine.process(calculated_df, detected_columns, source_name)
    st.session_state.active_package = package
    st.session_state.dossier_pdf = b""
    st.session_state.dossier_md = ""
    st.session_state["_dossier_sig"] = ""
    st.session_state.messages = [{"role": "assistant", "content": f"[OMNI-SYSTEM] {source_name} loaded — {package.summary['assets']} assets, {package.summary['records']} records, {package.summary['total_carbon']:.1f} tCO₂e total carbon.", "actions": package.summary.get("outlier_actions", [])[:2]}]
    st.session_state.session_carbon_g += carbon_tracker.estimate_upload_footprint(package.summary["records"], len(package.df.columns), source_name)


def render_message(message: dict[str, Any], index: int) -> None:
    role_class = "user" if message.get("role") == "user" else "assistant"
    label = "[DATA-QUERY]" if role_class == "user" else "[OMNI-SYSTEM]"
    content = html.escape(str(message.get('content', '')))

    st.markdown(
f"""<div style='display:flex;flex-direction:column;width:100%;'>
    <div class='message-card {role_class}'>
        <div class='role'>{label}</div>
        <div class='body'>{content}</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

    if message.get("plotly_fig") is not None:
        render_plot(message["plotly_fig"], key=f"chat_fig_{index}")

    for action_idx, action in enumerate(message.get("actions", [])):
        if st.button(action["label"], key=f"action_{index}_{action_idx}", use_container_width=True):
            if st.session_state.active_package is not None:
                updated, note = reg_engine.execute_action(action, st.session_state.active_package)
                st.session_state.active_package = updated
                st.session_state.messages.append({"role": "assistant", "content": note})
                st.rerun()


def render_command_snapshot(package) -> None:
    rankings = package.asset_rankings.reset_index(drop=True)
    best = rankings.iloc[0]["asset_id"] if not rankings.empty else "N/A"
    worst = rankings.iloc[-1]["asset_id"] if not rankings.empty else "N/A"
    st.markdown(
f"""<div class="cmd-snapshot">
    <div>
        <div class="cmd-tag">✦ OMNI-SYSTEM</div>
        <div class="cmd-title">Command Snapshot</div>
        <div class="cmd-desc">Active portfolio mapped deterministically. Best: <b style="color:#f0f6fc;">{best}</b> · Worst: <b style="color:#f0f6fc;">{worst}</b><br/>Use Control Room for trends, Anomaly Console for XAI, and Chat for live mutations.</div>
    </div>
    <div class="cmd-stat-box">
        <div class="cmd-stat-label">Active Source</div>
        <div class="cmd-stat-value">{package.source_name}</div>
        <div class="cmd-stat-rows">
            Records: <b>{package.summary['records']}</b><br/>
            Assets: <b>{package.summary['assets']}</b><br/>
            BRSR: <b class="green">Principle {package.summary['brsr_principle']}</b>
        </div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


# --- ALL ORIGINAL WORKSPACE FUNCTIONS BELOW (UNCHANGED IN LOGIC) ---

def render_dossier_console(package, forecast_df: pd.DataFrame) -> None:
    rankings = package.asset_rankings.reset_index(drop=True)
    anomaly_preview = package.anomaly_records.head(8).reset_index(drop=True)
    monthly = (
        package.df.assign(month_period=package.df["month"].dt.to_period("M").astype(str))
        .groupby("month_period", as_index=False)
        .agg(energy_kwh=("energy_kwh", "sum"), scope1=("Scope1_tCO2e", "sum"), scope2=("Scope2_tCO2e", "sum"), scope3=("Scope3_tCO2e", "sum"), anomalies=("Is_Anomaly", "sum"))
        .reset_index(drop=True)
    )
    dossier_tab_1, dossier_tab_2, dossier_tab_3 = st.tabs(
        ["[OMNI-SYSTEM] Executive Dossier", "[DATA-QUERY] Ledger + Rankings", "[CITED-RESEARCH] Standards + Memory"]
    )
    with dossier_tab_1:
        top_left, top_right = st.columns([1.15, 0.85], gap="large")
        with top_left:
            st.markdown("**Executive Summary**")
            st.markdown(
                f"- Total Carbon: `{package.summary['total_carbon']:.2f} tCO₂e`\n"
                f"- Scope 1 total: `{package.summary['scope1_total']:.2f} tCO2e`\n"
                f"- Scope 2 total: `{package.summary['scope2_total']:.2f} tCO2e`\n"
                f"- Scope 3 total: `{package.summary['scope3_total']:.2f} tCO2e`\n"
                f"- Total anomalies: `{package.summary['anomaly_count']}`\n"
                f"- Best performing asset: `{rankings.iloc[0]['asset_id'] if not rankings.empty else 'N/A'}`\n"
                f"- Worst performing asset: `{rankings.iloc[-1]['asset_id'] if not rankings.empty else 'N/A'}`"
            )
        with top_right:
            st.markdown("**AI Interpretation**")
            if not rankings.empty:
                st.markdown(
                    f"The portfolio is currently led by `{rankings.iloc[0]['asset_id']}` and most pressured by `{rankings.iloc[-1]['asset_id']}`. "
                    f"Electricity remains the dominant carbon source, so Scope 2 controls and renewable uplift are still the fastest route to compliance gains."
                )
    with dossier_tab_2:
        led_left, led_right = st.columns([1.0, 1.0], gap="large")
        with led_left:
            st.markdown("**Monthly Portfolio Ledger**")
            st.dataframe(monthly, use_container_width=True, hide_index=True)
        with led_right:
            st.markdown("**Asset Ranking Matrix**")
            st.dataframe(rankings.head(15), use_container_width=True, hide_index=True)
            st.markdown("**Anomaly Review**")
            st.dataframe(anomaly_preview, use_container_width=True, hide_index=True)
    with dossier_tab_3:
        std_left, std_right = st.columns([0.9, 1.1], gap="large")
        with std_left:
            st.markdown("**Standards Payload**")
            st.markdown("#### Regulatory Reference — Source of Truth")
        _ref = [
            ("Grid Emission Factor","0.727 kg CO₂/kWh","CEA CO₂ Baseline Ver 20.0 (Dec 2024)"),
            ("High-Speed Diesel (HSD)","2.68 kg CO₂/litre","IPCC 2006 AR4 — BEE/MoEFCC"),
            ("PNG / Natural Gas","2.02 kg CO₂/m³","GAIL / IPCC 2006"),
            ("R-410A GWP","2,088","India Cooling Action Plan (ICAP) / IPCC AR4"),
            ("R-134a GWP","1,430","ICAP / IPCC AR4"),
            ("Water — Bengaluru","0.80–1.00 kL/m²/yr","BEE Benchmarking Grade-A Offices"),
            ("Water — Mumbai","1.25–1.50 kL/m²/yr","BEE Benchmarking Grade-A Offices"),
            ("Indoor Design Temp","24°C ± 1°C","ISHRAE / BEE ECSBC 2024"),
            ("Chiller Water-Cooled","0.65–0.75 kW/TR","Industrial HVAC Audit Standard India"),
            ("EUI Baseline","140–180 kWh/m²/yr","BEE Star Rating Baseline Commercial"),
            ("Anomaly Precision","> 0.85","GreenLens SRD v3.0 UC-2"),
            ("Anomaly Recall","> 0.80","GreenLens SRD v3.0 UC-2"),
            ("Forecast MAPE","< 15%","GreenLens SRD v3.0 UC-3"),
        ]
        st.dataframe(pd.DataFrame(_ref, columns=["Parameter","Value / Benchmark","Official Source"]),
                     use_container_width=True, hide_index=True)
        with std_right:
            st.markdown("**Rolling Memory Chain-of-Custody**")
            st.code(st.session_state.rolling_context or "No compressed history yet.", language="text")
            st.markdown("**Forecast Preview**")
            st.dataframe(forecast_df.head(18).reset_index(drop=True), use_container_width=True, hide_index=True)


def build_monthly_ledger(package) -> pd.DataFrame:
    return (
        package.df.assign(month_period=package.df["month"].dt.to_period("M").astype(str))
        .groupby("month_period", as_index=False)
        .agg(
            energy_kwh=("energy_kwh", "sum"),
            scope1=("Scope1_tCO2e", "sum"),
            scope2=("Scope2_tCO2e", "sum"),
            scope3=("Scope3_tCO2e", "sum"),
            anomalies=("Is_Anomaly", "sum"),
            solar_kwh=("solar_kwh", "sum"),
            water_withdrawal_kl=("water_withdrawal_kl", "sum"),
        )
        .reset_index(drop=True)
    )


def build_scope_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        {"Scope": "Scope 1", "Definition": "Direct emissions from owned or controlled sources including diesel combustion and refrigerant-linked operational events.", "Primary Drivers": "Diesel litres, generator runtime, refrigerant loss proxies"},
        {"Scope": "Scope 2", "Definition": "Indirect emissions from purchased electricity, computed with the verified CEA Version 20 baseline factor.", "Primary Drivers": "Grid kWh, solar offset, HVAC load, occupancy-driven base load"},
        {"Scope": "Scope 3", "Definition": "Other indirect emissions and operational proxies across water, waste, hazardous waste, and procurement spend.", "Primary Drivers": "Water withdrawal, e-waste, hazardous waste, procurement spend"},
    ])


def build_metadata_matrix(package) -> pd.DataFrame:
    return pd.DataFrame([
        {"Field": "Generated At", "Value": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Field": "Source File", "Value": package.source_name},
        {"Field": "Reporting Period", "Value": f"{package.summary['reporting_start'].strftime('%Y-%m-%d')} to {package.summary['reporting_end'].strftime('%Y-%m-%d')}"},
        {"Field": "Assets", "Value": package.summary["assets"]},
        {"Field": "Records", "Value": package.summary["records"]},
        {"Field": "Framework", "Value": "SEBI BRSR Principle 6"},
        {"Field": "Scope 2 Benchmark", "Value": f"CEA baseline factor {DeterministicMath.CEA_FACTOR:.3f} kg CO2/kWh"},
        {"Field": "Operational Benchmark", "Value": "ECSBC-aligned energy intensity review"},
        {"Field": "AI Anomaly Precision Target", "Value": ">85%"},
    ])


def build_asset_categorization(package) -> pd.DataFrame:
    rankings = package.asset_rankings.reset_index(drop=True).copy()
    # Scope2_tCO2e and Scope3_tCO2e are already present in asset_rankings (from _rank_assets .agg).
    # Only merge if they are missing, to avoid _x/_y suffix conflicts.
    if "Scope2_tCO2e" not in rankings.columns:
        scope2_map = package.df.groupby("asset_id", as_index=False)["Scope2_tCO2e"].sum().reset_index(drop=True)
        rankings = rankings.merge(scope2_map, on="asset_id", how="left")
    if "Scope3_tCO2e" not in rankings.columns:
        scope3_map = package.df.groupby("asset_id", as_index=False)["Scope3_tCO2e"].sum().reset_index(drop=True)
        rankings = rankings.merge(scope3_map, on="asset_id", how="left")
    rankings["Category"] = rankings["Rank_Label"].replace({"Best Performing": "Leadership Asset", "Worst Performing": "Laggard Asset", "Core Performing": "Core Asset"})
    return rankings.rename(columns={"asset_id": "Asset", "Category": "Category", "Scope2_tCO2e": "Scope 2 tCO2e", "Scope3_tCO2e": "Scope 3 tCO2e", "Energy_Intensity_kWh_per_sqm": "Energy Intensity", "Anomalies": "Anomalies", "ESG_Score": "ESG Score", "BEE_Rating": "BEE Rating"})


def build_anomaly_review(package) -> pd.DataFrame:
    cols = ["month", "asset_id", "Anomaly_Signature", "energy_kwh", "water_withdrawal_kl", "diesel_litres", "Anomaly_Score"]
    if package.anomaly_records.empty:
        return pd.DataFrame(columns=cols)
    review = package.anomaly_records.loc[:, cols].copy().reset_index(drop=True)
    severity_rank = review["Anomaly_Score"].rank(pct=True, method="average")
    review["Severity"] = severity_rank.apply(lambda value: "Critical" if value >= 0.85 else ("High" if value >= 0.5 else "Elevated"))
    review["Month"] = pd.to_datetime(review["month"]).dt.strftime("%Y-%m")
    return review[["Month", "asset_id", "Severity", "Anomaly_Signature", "energy_kwh", "water_withdrawal_kl", "diesel_litres"]]


def build_shap_snapshot(package) -> pd.DataFrame:
    if package.latest_shap is None:
        return pd.DataFrame(columns=["Driver", "Impact"])
    items = list(package.latest_shap["values"].items())
    return pd.DataFrame(items, columns=["Driver", "Impact"]).reset_index(drop=True)


def build_forecast_outlook(forecast_df: pd.DataFrame) -> pd.DataFrame:
    return (
        forecast_df.groupby("month", as_index=False)[["Forecast_kWh", "Forecast_Scope2_tCO2e"]]
        .sum()
        .reset_index(drop=True)
        .assign(Forecast_Month=lambda frame: frame["month"].dt.strftime("%Y-%m"))
        .drop(columns=["month"])
    )


def build_advisory_actions(package) -> list[str]:
    recommendations = list(GREENLENS_SYSTEM_METADATA["Mitigation_Recommendations"])
    if not package.asset_rankings.empty:
        worst_asset = package.asset_rankings.iloc[-1]["asset_id"]
        recommendations.insert(0, f"Prioritize {worst_asset} for HVAC sequencing correction, diesel runtime governance, and envelope recommissioning before the next reporting window.")
    if package.summary["anomaly_count"] > 0:
        recommendations.append("Investigate anomaly clusters where negative fuel-volume deltas coincide with zero generator output to isolate leakage or logging faults.")
    return recommendations[:6]


def build_mapping_table(package) -> pd.DataFrame:
    return pd.DataFrame([{"Canonical Metric": canonical, "Detected Source Column": detected} for canonical, detected in package.detected_columns.items()]).reset_index(drop=True)


def build_data_quality_frame(package) -> pd.DataFrame:
    numeric_cols = package.df.select_dtypes(include="number").columns.tolist()
    rows = []
    for column in numeric_cols[:18]:
        rows.append({"Metric": column, "Missing After Clean": int(package.df[column].isna().sum()), "Median": round(float(package.df[column].median()), 4), "P95": round(float(package.df[column].quantile(0.95)), 4), "Max": round(float(package.df[column].max()), 4)})
    return pd.DataFrame(rows)


def render_control_header(package) -> None:
    rankings = package.asset_rankings.reset_index(drop=True)
    best_asset = rankings.iloc[0]["asset_id"] if not rankings.empty else "N/A"
    worst_asset = rankings.iloc[-1]["asset_id"] if not rankings.empty else "N/A"
    solar_pct = (package.summary['solar_total'] / max(package.summary['energy_total'], 1.0)) * 100
    st.markdown(
f"""<div class="ctrl-header">
    <div class="ctrl-cell">
        <div class="ctrl-label">Portfolio State</div>
        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1rem;font-weight:700;color:#f0f6fc;margin:6px 0 10px 0;">
            Deterministic compliance graph active across {package.summary['assets']} assets
        </div>
        <div class="ctrl-sub">
            Best: <b style="color:#c9d1d9;">{best_asset}</b> · Worst: <b style="color:#c9d1d9;">{worst_asset}</b><br/>
            Grid factor locked at {DeterministicMath.CEA_FACTOR:.3f} kg CO₂/kWh — CEA Version 20
        </div>
    </div>
    <div class="ctrl-cell" style="border-left:1px solid rgba(48,54,61,0.4);padding-left:16px;">
        <div class="ctrl-label">Renewable Position</div>
        <div class="ctrl-value green">{solar_pct:.2f}%</div>
        <div class="ctrl-sub">Solar as share of energy demand</div>
    </div>
    <div class="ctrl-cell" style="border-left:1px solid rgba(48,54,61,0.4);padding-left:16px;">
        <div class="ctrl-label">Anomaly Intensity</div>
        <div class="ctrl-value blue">{package.summary['anomaly_count']}</div>
        <div class="ctrl-sub">Isolation Forest detections &middot; 4% contamination (SRD v3.0)</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_portfolio_health(package) -> None:
    quality = build_data_quality_frame(package)
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        st.markdown("**Data Quality Envelope**")
        st.dataframe(quality, use_container_width=True, hide_index=True)
    with right:
        health_rows = pd.DataFrame([
            {"Metric": "Renewable Coverage", "Value": f"{(package.summary['solar_total'] / max(package.summary['energy_total'], 1.0)) * 100:.2f}%"},
            {"Metric": "Water Withdrawal", "Value": f"{package.summary['water_total']:.2f} kL"},
            {"Metric": "Total Energy", "Value": f"{package.summary['energy_total']:.2f} kWh"},
            {"Metric": "Anomaly Count", "Value": str(package.summary['anomaly_count'])},
            {"Metric": "Rolling Memory Lines", "Value": str(len(st.session_state.rolling_context.splitlines()))},
        ])
        st.markdown("**Operating Envelope**")
        st.dataframe(health_rows, use_container_width=True, hide_index=True)
        st.markdown("**Regulatory Basis**")
        st.markdown("The deterministic disclosures use verified Indian constants: CEA Version 20 for grid electricity, IPCC AR4 for diesel and refrigerants, and SEBI BRSR Principle 6 for reporting boundaries.")


def render_scope_and_metadata(package) -> None:
    scope_col, meta_col = st.columns([1.0, 1.0], gap="large")
    with scope_col:
        st.markdown("**Scope Definitions**")
        st.dataframe(build_scope_definitions(), use_container_width=True, hide_index=True)
    with meta_col:
        st.markdown("**Metadata Matrix**")
        st.dataframe(build_metadata_matrix(package), use_container_width=True, hide_index=True)


def render_executive_story(package) -> None:
    st.markdown("**Executive Summary**")
    for paragraph in reg_engine._synthesize_executive_summary(package):
        st.markdown(paragraph)


def render_asset_performance_matrix(package) -> None:
    categorization = build_asset_categorization(package)
    leader_col, laggard_col = st.columns([1.0, 1.0], gap="large")
    with leader_col:
        st.markdown("**Asset Categorization**")
        st.dataframe(categorization.head(12), use_container_width=True, hide_index=True)
    with laggard_col:
        st.markdown("**Laggard Focus Queue**")
        st.dataframe(categorization.sort_values(["Anomalies", "Energy Intensity"], ascending=[False, False]).head(12).reset_index(drop=True), use_container_width=True, hide_index=True)


def render_monthly_ledger_workspace(package, forecast_df: pd.DataFrame) -> None:
    ledger = build_monthly_ledger(package)
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        st.markdown("**Monthly Portfolio Ledger**")
        st.dataframe(ledger, use_container_width=True, hide_index=True)
    with right:
        forecast_view = build_forecast_outlook(forecast_df)
        st.markdown("**Forecast Outlook**")
        st.dataframe(forecast_view, use_container_width=True, hide_index=True)


def render_anomaly_and_xai_workspace(package, figures: dict[str, go.Figure]) -> None:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("**Anomaly Review**")
        st.dataframe(build_anomaly_review(package).head(20), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**SHAP Root-Cause Snapshot**")
        st.dataframe(build_shap_snapshot(package).head(12), use_container_width=True, hide_index=True)
        shap_fig = figures.get("SHAP Root-Cause Analysis")
        if shap_fig is not None:
            shap_fig.update_layout(height=480)
            render_plot(shap_fig, key="workspace_shap_root_cause")


def render_advisory_workspace(package) -> None:
    st.markdown("**Advisory Actions**")
    for action in build_advisory_actions(package):
        st.markdown(f"- {action}")


def render_mapping_workspace(package) -> None:
    map_left, map_right = st.columns([0.9, 1.1], gap="large")
    with map_left:
        st.markdown("**Dynamic Data Mapping**")
        st.dataframe(build_mapping_table(package), use_container_width=True, hide_index=True)
    with map_right:
        st.markdown("**System Metadata Schema**")
        st.markdown("**Regulatory Constants**")
        st.markdown(
            f"- Grid factor: `{DeterministicMath.CEA_FACTOR:.3f} kg CO₂/kWh`  \u2014  CEA v20\n"
            "- Diesel factor: `2.68 kg CO₂/L`  \u2014  IPCC AR4\n"
            "- R-410A GWP: `2,088`  \u2014  ICAP/IPCC AR4\n"
            "- R-134a GWP: `1,430`  \u2014  ICAP/IPCC AR4\n"
            "- Chiller (water-cooled): `0.65\u20130.75 kW/TR`  \u2014  HVAC Audit Std India\n"
            "- EUI baseline: `140\u2013180 kWh/m\u00b2/yr`  \u2014  BEE Star Rating\n"
            "- Anomaly precision: `>0.85`  \u2014  GreenLens SRD v3.0 UC-2"
        )


def render_audit_workspace() -> None:
    st.markdown("**Official Audit Trail**")
    if st.session_state.messages:
        for idx, message in enumerate(st.session_state.messages[-18:]):
            role_class = "user" if message.get("role") == "user" else "assistant"
            label = "[DATA-QUERY]" if role_class == "user" else "[OMNI-SYSTEM]"
            content = html.escape(str(message.get("content", "")))
            st.markdown(
f"""<div style='display:flex;flex-direction:column;width:100%;margin-bottom:12px;'>
    <div class='message-card {role_class}'>
        <div class='role'>{label}</div>
        <div class='body'>{content}</div>
    </div>
</div>""",
                unsafe_allow_html=True,
            )
            if message.get("plotly_fig") is not None:
                render_plot(message["plotly_fig"], key=f"audit_fig_{idx}")
    else:
        st.info("No chat history yet.")


def render_regulatory_reference_workspace() -> None:
    st.markdown("**Benchmark & Standards — Source of Truth**")
    st.caption("Every constant below is aligned with its official Indian regulatory citation. Sources: CEA v20 (Dec 2024), IPCC AR4, ICAP, BEE ECSBC 2024, ISHRAE, SEBI BRSR (May 2021).")
    _master_ref = [
        ("Emission Factors", "Grid Emission Factor (Average)", "0.727 kg CO₂/kWh", "CEA CO₂ Baseline Database (Ver 20.0, Dec 2024)"),
        ("Emission Factors", "High-Speed Diesel (HSD)", "2.68 kg CO₂/Litre", "IPCC 2006 AR4 — adopted by BEE/MoEFCC"),
        ("Emission Factors", "Refrigerant GWP R-410A", "2,088", "India Cooling Action Plan (ICAP) / IPCC AR4"),
        ("Emission Factors", "Refrigerant GWP R-134a", "1,430", "India Cooling Action Plan (ICAP) / IPCC AR4"),
        ("Emission Factors", "Piped Natural Gas (PNG)", "2.02 kg CO₂/m³", "GAIL / IPCC 2006"),
        ("Water Intensity", "Bengaluru (Temperate)", "0.80 – 1.00 kL/m²/year", "BEE Performance Benchmarking (Grade A Offices)"),
        ("Water Intensity", "Mumbai (Warm & Humid)", "1.25 – 1.50 kL/m²/year", "BEE Performance Benchmarking (Grade A Offices)"),
        ("Building Physics", "Max SHGC (Fenestration)", "≤ 0.25", "Energy Conservation & Sustainable Building Code (ECSBC 2024)"),
        ("Building Physics", "Max U-Factor (Wall)", "0.40 W/m²K", "ECSBC 2024"),
        ("Building Physics", "Indoor Design Condition", "24°C ± 1°C", "ISHRAE / BEE ECSBC 2024 Standard"),
        ("Building Physics", "Chiller Efficiency (Water-Cooled)", "0.65 – 0.75 kW/TR", "Industrial HVAC Audit Standard (India)"),
        ("Building Physics", "Baseload Load Ratio", "28% – 32%", "Indian IT Park Load Profile Research (Commercial)"),
        ("Building Physics", "Thermal Conversion", "1 TR = 3.517 kW", "ASHRAE / BEE standard"),
        ("AI Performance", "Anomaly Precision", "> 0.85", "GreenLens SRD v3.0 Acceptance Criteria (UC-2)"),
        ("AI Performance", "Anomaly Recall", "> 0.80", "GreenLens SRD v3.0 Acceptance Criteria (UC-2)"),
        ("AI Performance", "Forecast Accuracy (MAPE)", "< 15%", "GreenLens SRD v3.0 Acceptance Criteria (UC-3)"),
        ("AI Performance", "Field Extraction Accuracy", "> 85%", "GreenLens SRD v3.0 Acceptance Criteria (UC-1)"),
        ("BRSR Indicators", "Essential Indicators (EI)", "EI-1 to EI-10", "SEBI BRSR Principle 6 Mandatory Disclosures"),
        ("BRSR Indicators", "Scope 3 Weighting", "15% – 20% of Env Score", "SEBI Leadership Indicators / Big 4 Rating Matrix"),
        ("BRSR Indicators", "Energy Intensity (EUI)", "140 – 180 kWh/m²/year", "BEE Star Rating Baseline (Commercial)"),
        ("Anomaly Math", "Refrigerant Leak Threshold", "> 12% Energy Spike", "Physics-based Isolation Forest Signature"),
        ("Anomaly Math", "Diesel Theft Threshold", "> 0.5 L/hr at Idle", "DG Set Telemetry Outlier Analysis"),
    ]
    import pandas as _pd2
    _ref_df = _pd2.DataFrame(_master_ref, columns=["Category", "Metric / Parameter", "Value / Benchmark", "Official Source / Citation"])
    st.dataframe(_ref_df, use_container_width=True, hide_index=True)
    st.caption(
        "Verification: (1) Regulatory Alignment — every emission constant aligned with CEA or IPCC Tier 1 defaults as required for ‘Reasonable Assurance’ in Indian ESG audits. "
        "(2) Building Science — thermal benchmarks from BEE ECSBC 2024, the current statutory requirement for high-granularity energy modelling in India. "
        "(3) Audit Readiness — Precision/Recall/MAPE derived from the Software Requirements Document (SRD) to ensure AI outputs are defensible during Big 4 third-party certification."
    )


def render_score_methodology_workspace(package) -> None:
    score_left, score_right = st.columns([1.05, 0.95], gap="large")
    with score_left:
        st.markdown("**CUESG Energy + Emission Performance Methodology**")
        st.markdown("Performance signals are computed deterministically: energy intensity (kWh/m²), anomaly detection rate (Isolation Forest 4%), solar integration ratio (BEE benchmark), and Scope 3 proxy coverage (SEBI BRSR P6). All signals are recalculated after every ingestion event, OCR append, mutation, or scenario run.")
        methodology = pd.DataFrame([
            {"Component": "Energy Intensity", "Current Signal": round(float(package.df['Energy_Intensity_kWh_per_sqm'].mean()), 4)},
            {"Component": "Anomaly Rate", "Current Signal": round(float(package.df['Is_Anomaly'].mean()), 4)},
            {"Component": "Solar Integration", "Current Signal": round(float(package.df['Solar_Integration_Ratio'].mean()), 4)},
            {"Component": "Scope 3 Efficiency", "Current Signal": round(1 - min(package.summary['scope3_total'] / max(package.summary['total_carbon'], 1.0), 1.0), 4)},
        ])
        st.dataframe(methodology, use_container_width=True, hide_index=True)
    with score_right:
        rating_counts = package.asset_rankings["BEE_Rating"].astype(str).value_counts().reset_index()
        rating_counts.columns = ["BEE Rating", "Assets"]
        bee_mix = px.bar(rating_counts, x="BEE Rating", y="Assets", color="BEE Rating", color_discrete_sequence=["#00ff7f", "#58a6ff", "#79c0ff", "#d29922", "#f85149"])
        bee_mix.update_layout(**fig_layout("BEE Rating Mix"))
        render_plot(bee_mix, key="bee_rating_mix")


def render_heatmap_workspace(package) -> None:
    st.markdown("**Asset Heatmap Matrix**")
    heatmap_df = (
        package.df.assign(month_label=package.df["month"].dt.strftime("%Y-%m"))
        .pivot_table(index="asset_id", columns="month_label", values="Energy_Intensity_kWh_per_sqm", aggfunc="mean")
        .fillna(0.0)
        .reset_index()
    )
    if not heatmap_df.empty:
        heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_df.drop(columns=["asset_id"]).values,
            x=heatmap_df.columns[1:],
            y=heatmap_df["asset_id"],
            colorscale=[[0.0, "#060a0f"], [0.35, "#58a6ff"], [0.7, "#d29922"], [1.0, "#f85149"]],
        ))
        heatmap.update_layout(**fig_layout("Energy Intensity Heatmap"))
        render_plot(heatmap, key="energy_intensity_heatmap")


def render_cross_signal_workspace(package) -> None:
    st.markdown("**Cross-Signal Correlation Arena**")
    sample = package.df[["energy_kwh", "diesel_litres", "solar_kwh", "water_withdrawal_kl", "hvac_load_kw", "outdoor_temp_c", "occupancy_percent", "Total_tCO2e"]].corr(numeric_only=True).reset_index()
    st.dataframe(sample, use_container_width=True, hide_index=True)


def render_climate_benchmark_workspace(package) -> None:
    benchmark = (
        package.df.groupby("Climate_Zone", as_index=False)
        .agg(Avg_Energy_Intensity=("Energy_Intensity_kWh_per_sqm", "mean"), Avg_Annualized_Water=("Annualized_Water_Intensity", "mean"), Avg_kW_per_TR=("kW_per_TR", "mean"), Assets=("asset_id", "nunique"))
        .reset_index(drop=True)
    )
    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        benchmark_fig = go.Figure()
        benchmark_fig.add_bar(x=benchmark["Climate_Zone"], y=benchmark["Avg_Energy_Intensity"], name="Energy Intensity", marker_color="#58a6ff")
        benchmark_fig.add_scatter(x=benchmark["Climate_Zone"], y=benchmark["Avg_Annualized_Water"], name="Annualized Water", mode="lines+markers", line=dict(color="#00ff7f", width=3), yaxis="y2")
        benchmark_fig.update_layout(**fig_layout("Climate Benchmark Drift"), yaxis=dict(title="Energy Intensity"), yaxis2=dict(title="Water Intensity", overlaying="y", side="right"))
        render_plot(benchmark_fig, key="climate_benchmark_drift")
    with right:
        st.dataframe(benchmark, use_container_width=True, hide_index=True)


def render_scenario_library(package) -> None:
    st.markdown("**Scenario Library**")
    scenarios = pd.DataFrame([
        {"Scenario": "Solar Expansion 50kW", "Command": "What happens if we install 50kW of solar next month?", "Intent": "Scope 2 and ROI projection"},
        {"Scenario": "Solar Expansion 100kW", "Command": "What happens if we install 100kW of solar next month?", "Intent": "Aggressive renewable uplift"},
        {"Scenario": "HVAC Setpoint -2C", "Command": "What happens if we reduce HVAC setpoint by 2 C?", "Intent": "Cooling load stress test"},
        {"Scenario": "Combined Solar and HVAC", "Command": "What happens if we install 100kW of solar and reduce setpoint by 2 C?", "Intent": "Multi-variable what-if analysis"},
        {"Scenario": "Diesel Stress", "Command": "Set BLD-MUM-001 diesel consumption to 800 liters for March", "Intent": "Scope 1 shock analysis"},
    ])
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    button_cols = st.columns(5, gap="small")
    scenario_prompts = scenarios["Command"].tolist()
    for idx, col in enumerate(button_cols):
        with col:
            if st.button(f"Run S{idx + 1}", key=f"scenario_lib_{idx}", use_container_width=True):
                queue_prompt(scenario_prompts[idx])


def render_memory_workspace() -> None:
    st.markdown("**Rolling Context Memory**")
    mem_left, mem_right = st.columns([0.9, 1.1], gap="large")
    with mem_left:
        memory_rows = []
        for idx, message in enumerate(st.session_state.messages[-20:]):
            memory_rows.append({"Index": idx + 1, "Role": message.get("role", "assistant"), "Excerpt": str(message.get("content", ""))[:160], "Has Chart": bool(message.get("plotly_fig") is not None), "Has Action": bool(message.get("actions"))})
        st.dataframe(pd.DataFrame(memory_rows), use_container_width=True, hide_index=True)
    with mem_right:
        st.markdown("**Compressed System Context**")
        st.code(st.session_state.rolling_context or "No compressed context yet.", language="text")


def render_source_citation_workspace() -> None:
    st.markdown("**Source Citation Engine**")
    citation_rows = pd.DataFrame([
        {"Standard": "SEBI BRSR Principle 6", "Usage": "Disclosure categorization and essential indicator framing"},
        {"Standard": "CEA Version 20", "Usage": "Grid electricity emission factor for Scope 2"},
        {"Standard": "IPCC AR4", "Usage": "Diesel and refrigerant GWP accounting constants"},
        {"Standard": "ICAP", "Usage": "Refrigerant global warming potential references"},
        {"Standard": "ECSBC 2024", "Usage": "Envelope and operating baseline context for HVAC reasoning"},
    ])
    st.dataframe(citation_rows, use_container_width=True, hide_index=True)


def render_asset_command_cards(package) -> None:
    st.markdown("**Asset Command Cards**")
    top_assets = package.asset_rankings.head(6).reset_index(drop=True)
    cols = st.columns(3, gap="large")
    for idx, row in top_assets.iterrows():
        with cols[idx % 3]:
            st.markdown(
f"""<div class="asset-card">
    <div class="asset-name">{row['asset_id']}</div>
    <div class="asset-meta">
        BEE Rating: <b>{row['BEE_Rating']}</b><br/>
        Energy Intensity: <b>{row['Energy_Intensity_kWh_per_sqm']:.3f} kWh/m²</b><br/>
        Scope 2: <b>{row.get('Scope2_tCO2e', 0.0):.2f} tCO₂e</b><br/>
        Anomalies: <b>{int(row['Anomalies'])}</b>
    </div>
</div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"Explain {row['asset_id']}", key=f"asset_explain_{idx}", use_container_width=True):
                queue_prompt(f"Analyse asset {row['asset_id']} and explain its current score.")


def render_operating_ratio_workspace(package) -> None:
    st.markdown("**Operating Ratios**")
    ratio_df = pd.DataFrame([
        {"Ratio": "Solar / Energy", "Value": round(float(package.summary["solar_total"] / max(package.summary["energy_total"], 1.0)), 4)},
        {"Ratio": "Scope 1 / Total", "Value": round(float(package.summary["scope1_total"] / max(package.summary["total_carbon"], 1.0)), 4)},
        {"Ratio": "Scope 2 / Total", "Value": round(float(package.summary["scope2_total"] / max(package.summary["total_carbon"], 1.0)), 4)},
        {"Ratio": "Scope 3 / Total", "Value": round(float(package.summary["scope3_total"] / max(package.summary["total_carbon"], 1.0)), 4)},
        {"Ratio": "Anomalies / Records", "Value": round(float(package.summary["anomaly_count"] / max(package.summary["records"], 1.0)), 4)},
        {"Ratio": "Water Benchmark Breaches / Records", "Value": round(float(package.df["Water_Benchmark_Breach"].sum() / max(package.summary["records"], 1.0)), 4)},
    ])
    st.dataframe(ratio_df, use_container_width=True, hide_index=True)


def render_portfolio_timeline_workspace(package) -> None:
    st.markdown("**Portfolio Timeline**")
    timeline = build_monthly_ledger(package).copy()
    timeline_fig = go.Figure()
    timeline_fig.add_scatter(x=timeline["month_period"], y=timeline["scope2"], mode="lines+markers", name="Scope 2", line=dict(color="#58a6ff", width=2.5))
    timeline_fig.add_scatter(x=timeline["month_period"], y=timeline["scope1"], mode="lines+markers", name="Scope 1", line=dict(color="#f78166", width=2))
    timeline_fig.add_scatter(x=timeline["month_period"], y=timeline["scope3"], mode="lines+markers", name="Scope 3", line=dict(color="#00ff7f", width=2))
    timeline_fig.update_layout(**fig_layout("Portfolio Carbon Timeline"))
    render_plot(timeline_fig, key="portfolio_carbon_timeline")


def render_signal_decomposition_workspace(package) -> None:
    st.markdown("**Signal Decomposition Grid**")
    decomp = (
        package.df.groupby("asset_id", as_index=False)
        .agg(Grid_Energy=("energy_kwh", "sum"), Diesel=("diesel_litres", "sum"), Solar=("solar_kwh", "sum"), Water=("water_withdrawal_kl", "sum"), Waste=("ewaste_kg", "sum"), Spend=("procurement_spend_inr", "sum"))
        .reset_index(drop=True)
    )
    st.dataframe(decomp.head(25), use_container_width=True, hide_index=True)


def render_top_bottom_workspace(package) -> None:
    top = package.asset_rankings.head(5).reset_index(drop=True)
    bottom = package.asset_rankings.tail(5).reset_index(drop=True)
    top_bottom = st.columns(2, gap="large")
    with top_bottom[0]:
        st.markdown("**Top 5 Buildings**")
        st.dataframe(top, use_container_width=True, hide_index=True)
    with top_bottom[1]:
        st.markdown("**Bottom 5 Buildings**")
        st.dataframe(bottom, use_container_width=True, hide_index=True)


def render_chat_analytics_workspace() -> None:
    st.markdown("**Chat Analytics**")
    rows = []
    for message in st.session_state.messages:
        rows.append({"Role": message.get("role", ""), "Characters": len(str(message.get("content", ""))), "Has Plot": bool(message.get("plotly_fig") is not None), "Has Agent Action": bool(message.get("actions"))})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        summary = frame.groupby("Role", as_index=False).agg(Messages=("Role", "count"), Avg_Chars=("Characters", "mean")).reset_index(drop=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)


def render_compliance_checklist_workspace(package) -> None:
    st.markdown("**Compliance Checklist**")
    checklist = pd.DataFrame([
        {"Indicator": "Total energy consumed", "Status": "Ready", "Evidence": f"{package.summary['energy_total']:.2f} kWh"},
        {"Indicator": "Scope 1 emissions", "Status": "Ready", "Evidence": f"{package.summary['scope1_total']:.2f} tCO2e"},
        {"Indicator": "Scope 2 emissions", "Status": "Ready", "Evidence": f"{package.summary['scope2_total']:.2f} tCO2e"},
        {"Indicator": "Water withdrawal", "Status": "Ready", "Evidence": f"{package.summary['water_total']:.2f} kL"},
        {"Indicator": "Solid waste generated", "Status": "Ready", "Evidence": f"{package.df['ewaste_kg'].sum():.2f} kg"},
        {"Indicator": "Hazardous waste generated", "Status": "Ready", "Evidence": f"{package.df['hazardous_waste_kg'].sum():.2f} kg"},
        {"Indicator": "Scope 3 proxy coverage", "Status": "Ready", "Evidence": f"{package.summary['scope3_total']:.2f} tCO2e"},
    ])
    st.dataframe(checklist, use_container_width=True, hide_index=True)


def render_dossier_text_workspace(package, forecast_df: pd.DataFrame) -> None:
    st.markdown("**Rendered Dossier Text**")
    dossier_text = reg_engine.build_markdown_report(package, forecast_df, st.session_state.messages, st.session_state.rolling_context)
    st.markdown(dossier_text)


def render_command_reference_workspace() -> None:
    st.markdown("**Command Reference**")
    commands = pd.DataFrame([
        {"Command Pattern": "List the 5 most sustainable buildings", "Engine Response": "Ranked leaderboard by energy intensity and BEE rating"},
        {"Command Pattern": "Why did our score drop?", "Engine Response": "SHAP waterfall with plain-English explanation"},
        {"Command Pattern": "Set BLD-01 diesel consumption to 800 liters", "Engine Response": "Live dataframe mutation and recalc"},
        {"Command Pattern": "What happens if we install 100kW of solar next month?", "Engine Response": "Forecast, carbon delta, ROI"},
        {"Command Pattern": "Export the official compliance dossier", "Engine Response": "Markdown and PDF dossier generation"},
        {"Command Pattern": "Compare top and bottom buildings", "Engine Response": "Comparative portfolio analytics"},
    ])
    st.dataframe(commands, use_container_width=True, hide_index=True)
    st.markdown("Every command issued through the chat feed is preserved in `st.session_state.messages`, compressed into rolling memory when needed, and appended to the PDF audit trail when the dossier is exported.")


def render_system_runtime_workspace(package) -> None:
    st.markdown("**System Runtime**")
    # Detect available OCR engines
    ocr_engines = []
    try:
        import pdfplumber
        ocr_engines.append("pdfplumber")
    except ImportError:
        pass
    try:
        import fitz
        ocr_engines.append("PyMuPDF")
    except ImportError:
        pass
    try:
        import pytesseract
        ocr_engines.append("Tesseract")
    except ImportError:
        pass
    try:
        import reportlab
        rl_status = "ReportLab ✓"
    except ImportError:
        rl_status = "ReportLab ✗ (plain-text PDF)"

    gemini_status = "✓ Gemini 1.5 Flash Active" if bool(os.getenv("GEMINI_API_KEY", "")) else "✗ Not set — add GEMINI_API_KEY to .env for AI OCR"
    openai_status = "✓ Active" if bool(os.getenv("OPENAI_API_KEY", "")) else "✗ Not set"

    runtime = pd.DataFrame([
        {"Signal": "Control Room Mode",      "Value": str(st.session_state.control_room)},
        {"Signal": "Active Source",          "Value": package.source_name},
        {"Signal": "Queued Prompt",          "Value": st.session_state.queued_prompt or "None"},
        {"Signal": "Pending OCR Payload",    "Value": "Yes" if st.session_state.pending_bill is not None else "No"},
        {"Signal": "Messages in Memory",     "Value": len(st.session_state.messages)},
        {"Signal": "Detected Columns",       "Value": len(package.detected_columns)},
        {"Signal": "Session Carbon (gCO2e)", "Value": f"{st.session_state.session_carbon_g:.4f}"},
        {"Signal": "OCR Engines Available",  "Value": ", ".join(ocr_engines) if ocr_engines else "Regex only (offline)"},
        {"Signal": "PDF Engine",             "Value": rl_status},
        {"Signal": "GEMINI_API_KEY",         "Value": gemini_status},
        {"Signal": "OPENAI_API_KEY",         "Value": openai_status},
    ])
    st.dataframe(runtime, use_container_width=True, hide_index=True)


def render_report_status_workspace(package) -> None:
    st.markdown("**Report Status**")
    report_status = pd.DataFrame([
        {"Artifact": "PDF Report", "State": "Ready" if bool(st.session_state.dossier_pdf) else "Pending"},
        {"Artifact": "Markdown Dossier", "State": "Ready" if bool(st.session_state.dossier_md) else "Pending"},
        {"Artifact": "Audit History", "State": "Ready" if bool(st.session_state.messages) else "Pending"},
        {"Artifact": "Detected Source Map", "State": f"{len(package.detected_columns)} fields"},
        {"Artifact": "Active Source", "State": package.source_name},
    ])
    st.dataframe(report_status, use_container_width=True, hide_index=True)


def render_detailed_dossier_workspace(package, forecast_df: pd.DataFrame, figures: dict[str, go.Figure]) -> None:
    st.markdown("## Dossier Command Surface")
    top_tabs = st.tabs([
        "[OMNI-SYSTEM] Executive Deck",
        "[DATA-QUERY] Performance Matrix",
        "[XAI-ALERT] Anomaly + Forecast",
        "[CITED-RESEARCH] Compliance Kernel",
        "[OMNI-SYSTEM] Audit Ledger",
        "[DATA-QUERY] Deep Ops",
        "[CITED-RESEARCH] Scenario Library",
        "[OMNI-SYSTEM] Memory + Chat",
        "[DATA-QUERY] Full Dossier",
    ])
    with top_tabs[0]:
        render_scope_and_metadata(package)
        render_executive_story(package)
        render_score_methodology_workspace(package)
    with top_tabs[1]:
        render_asset_performance_matrix(package)
        render_monthly_ledger_workspace(package, forecast_df)
    with top_tabs[2]:
        render_anomaly_and_xai_workspace(package, figures)
        render_advisory_workspace(package)
    with top_tabs[3]:
        render_mapping_workspace(package)
        render_regulatory_reference_workspace()
    with top_tabs[4]:
        render_audit_workspace()
    with top_tabs[5]:
        render_portfolio_health(package)
        render_heatmap_workspace(package)
        render_cross_signal_workspace(package)
        render_climate_benchmark_workspace(package)
        render_portfolio_timeline_workspace(package)
        render_signal_decomposition_workspace(package)
        render_top_bottom_workspace(package)
    with top_tabs[6]:
        render_scenario_library(package)
        render_asset_command_cards(package)
        render_operating_ratio_workspace(package)
        render_compliance_checklist_workspace(package)
    with top_tabs[7]:
        render_memory_workspace()
        render_chat_analytics_workspace()
        render_source_citation_workspace()
    with top_tabs[8]:
        render_dossier_text_workspace(package, forecast_df)
        render_command_reference_workspace()
        render_system_runtime_workspace(package)
        render_report_status_workspace(package)



# ── Gemini Vision OCR Helper ──────────────────────────────────────────────────
def _parse_bill_gemini(file_bytes: bytes, mime_type: str, is_pdf: bool = False) -> dict:
    """
    Extract ESG metrics from any utility bill using Gemini Vision API.
    Falls back gracefully to regex/pdfplumber if API key not set.
    """
    import base64, json as _json, re as _re

    GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

    if GEMINI_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Encode file
            b64 = base64.b64encode(file_bytes).decode()

            prompt = """You are an ESG data extraction engine. Extract the following fields from this utility bill image.
Return ONLY a JSON object with these exact keys (use null if not found):
{
  "energy_kwh": <number or null>,
  "diesel_litres": <number or null>,
  "water_withdrawal_kl": <number or null>,
  "solar_kwh": <number or null>,
  "utility_cost_inr": <number or null>,
  "bill_month": "<YYYY-MM or null>",
  "vendor_name": "<string or null>",
  "account_id": "<string or null>",
  "asset_id": "<string or null>",
  "meter_reading_start": <number or null>,
  "meter_reading_end": <number or null>
}
Extract only from what is clearly visible. Do not guess. Return valid JSON only."""

            response = model.generate_content([
                prompt,
                {"mime_type": mime_type if not is_pdf else "application/pdf", "data": b64}
            ])

            raw = response.text.strip()
            # Strip markdown code fences if present
            raw = _re.sub(r"^```[a-z]*\n?", "", raw)
            raw = _re.sub(r"\n?```$", "", raw)
            extracted = _json.loads(raw)

            # Score confidence based on how many key fields were found
            key_fields = ["energy_kwh", "diesel_litres", "water_withdrawal_kl", "utility_cost_inr"]
            found = sum(1 for k in key_fields if extracted.get(k) is not None)
            confidence = 0.70 + (found / len(key_fields)) * 0.30

            extracted["_confidence"] = round(confidence, 2)
            extracted["_ocr_engine"] = "Gemini 1.5 Flash Vision"
            extracted["_status"] = f"Gemini Vision extracted {found}/{len(key_fields)} key fields"
            extracted["_flag_for_review"] = confidence < 0.90
            extracted["_auto_ingest"] = confidence >= 0.75
            return extracted

        except Exception as _gem_e:
            # Fall through to regex fallback
            pass

    # ── Fallback: use existing reg_engine parsers ──
    if is_pdf:
        result = reg_engine.parse_utility_bill_pdf(file_bytes)
    else:
        result = reg_engine.parse_utility_bill_image(file_bytes, mime_type)

    # Ensure _auto_ingest is set based on confidence
    conf = float(result.get("_confidence", 0.0))
    result["_auto_ingest"] = conf >= 0.75
    result["_flag_for_review"] = 0.75 <= conf < 0.90
    if not GEMINI_KEY:
        result["_status"] = (result.get("_status") or "") + " | Set GEMINI_API_KEY env var for AI-powered extraction"
    return result

# --- MAIN LAYOUT ---

main_col = st.container()

with main_col:
    # Action bar: Generate Report | Analysis | Reset
    ab_left, ab_mid, ab_right = st.columns([3, 3, 1], gap="small")
    if ab_left.button("⬡ Generate Compliance Report", use_container_width=True):
        _pkg_btn = st.session_state.active_package
        if _pkg_btn is None:
            st.warning("Load a dataset first.")
        else:
            st.session_state["_trigger_pdf_build"] = True
            
    if ab_mid.button("◎ Run Anomaly + Forecast Analysis", use_container_width=True):
        queue_prompt("Run full anomaly detection and 12-month forecast. Summarise key findings.")
        
    if ab_right.button("↺ Reset", use_container_width=True):
        for key in ["messages", "active_package", "pending_bill", "rolling_context",
                    "dossier_md", "dossier_pdf", "cached_forecast_df"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if not st.session_state.control_room:
        st.markdown(
            """<div class="hero-container">
    <div class="hero-title">CUESG</div>
    <div class="hero-copy">Drop in a messy spreadsheet — or hit ⚡ Load Demo Data to start instantly. Tracks carbon across all three scopes, parses utility bills via OCR, flags anomalies before they become audit issues, and runs what-if scenarios against SEBI BRSR Principle 6. Built for Indian commercial portfolios.</div>
</div>""",
            unsafe_allow_html=True,
        )

        # Upload controls under the title
        up_left, up_right = st.columns([1, 1], gap="medium")
        with up_left:
            dataset_file = st.file_uploader("Upload ESG dataset", type=["csv", "xlsx", "xls", "pdf"])
            if dataset_file is not None:
                signature = f"{dataset_file.name}:{dataset_file.size}"
                if signature != st.session_state.active_signature:
                    try:
                        activate_portfolio(read_uploaded_file(dataset_file), dataset_file.name)
                        st.session_state.active_signature = signature
                    except Exception as exc:
                        st.error(f"Could not load dataset: {exc}")
                        
            demo_col, master_col = st.columns(2, gap="small")
            with demo_col:
                if st.button("⚡ Load Demo Data", use_container_width=True, help="Generate synthetic 8-asset, 24-month Indian portfolio — no file needed"):
                    demo_df = generate_synthetic_demo_data()
                    activate_portfolio(demo_df, "CUESG-Demo-Portfolio (Synthetic)")
                    st.session_state.active_signature = "demo"
                    
            with master_col:
                if st.button("Load Master Dataset", use_container_width=True):
                    master = load_master_dataset()
                    if not master.empty:
                        activate_portfolio(master, "master_esg_data")
                    else:
                        st.warning("master_esg_data.xlsx / .csv not found. Use ⚡ Load Demo Data instead.")

        with up_right:
            st.markdown(
                "<div style='font-family:Manrope,sans-serif;font-size:0.65rem;"
                "color:#4d5a6a;letter-spacing:0.12em;text-transform:uppercase;"
                "font-weight:700;margin-bottom:6px;'>✦ Utility Bill Parser</div>",
                unsafe_allow_html=True,
            )
            bill_type = st.radio("Bill format", ["Image (PNG/JPG)", "PDF Bill"], horizontal=True, label_visibility="collapsed", key="bill_type_radio")

            if bill_type == "Image (PNG/JPG)":
                bill_image = st.file_uploader(
                    "Upload bill image (BESCOM, MSEDCL, tanker receipt…)",
                    type=["png", "jpg", "jpeg", "webp"], key="bill_image"
                )
                if bill_image is not None:
                    if st.button("🔍 Parse Bill OCR", use_container_width=True, key="parse_img_btn"):
                        with st.spinner("Running CUESG Omni-Parser (Gemini Vision + regex fallback)…"):
                            try:
                                result = _parse_bill_gemini(bill_image.getvalue(), bill_image.type or "image/png", is_pdf=False)
                                st.session_state.pending_bill = result
                            except Exception as _ocr_e:
                                st.error(f"OCR error: {_ocr_e}")
                                result = reg_engine.parse_utility_bill_image(bill_image.getvalue(), bill_image.type or "image/png")
                                st.session_state.pending_bill = result
            else:
                bill_pdf = st.file_uploader(
                    "Upload utility bill PDF (BESCOM, Tata Power, Water board…)",
                    type=["pdf"], key="bill_pdf"
                )
                if bill_pdf is not None:
                    if st.button("🔍 Parse PDF Bill", use_container_width=True, key="parse_pdf_btn"):
                        with st.spinner("Running CUESG PDF Parser (Gemini Vision + pdfplumber fallback)…"):
                            try:
                                result = _parse_bill_gemini(bill_pdf.getvalue(), "application/pdf", is_pdf=True)
                                st.session_state.pending_bill = result
                            except Exception as _pdf_e:
                                st.error(f"PDF OCR error: {_pdf_e}")
                                result = reg_engine.parse_utility_bill_pdf(bill_pdf.getvalue())
                                st.session_state.pending_bill = result

        # ── OCR Result Display ────────────────────────────────────────────
        if st.session_state.pending_bill is not None:
            bill = st.session_state.pending_bill
            conf = float(bill.get("_confidence", 0.0))
            status = str(bill.get("_status", ""))
            engine = str(bill.get("_ocr_engine", "unknown"))
            flag = bool(bill.get("_flag_for_review", False))
            auto_ingest = bool(bill.get("_auto_ingest", False))

            # Confidence badge
            if conf >= 0.90:
                badge_color, badge_label = "#00ff7f", "✓ AUTO-ACCEPT"
            elif conf >= 0.75:
                badge_color, badge_label = "#d29922", "⚠ REVIEW FLAGGED"
            else:
                badge_color, badge_label = "#f85149", "✗ MANUAL ENTRY"

            st.markdown(
                f"<div style='font-family:Manrope,sans-serif;font-size:0.68rem;"
                f"background:rgba(13,17,23,0.8);border:1px solid {badge_color}33;"
                f"border-left:3px solid {badge_color};padding:10px 14px;"
                f"border-radius:8px;margin:8px 0;'>"
                f"<span style='color:{badge_color};font-weight:700;'>{badge_label}</span>"
                f"<span style='color:#4d5a6a;margin-left:10px;'>Confidence: {conf:.2f} | Engine: {engine}</span>"
                f"<br/><span style='color:#8b949e;font-size:0.62rem;'>{status}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if auto_ingest:
                # Show clean extracted fields
                display_keys = ["energy_kwh", "diesel_litres", "water_withdrawal_kl", "utility_cost_inr", "solar_kwh", "bill_month", "vendor_name", "account_id"]
                clean_bill = {k: v for k, v in bill.items() if k in display_keys and v is not None}
                if clean_bill:
                    st.json(clean_bill)
                if flag:
                    st.warning("⚠ Yellow flag: confidence 0.75–0.89. This record will be marked for monthly audit review.")
                if st.button("✅ Append to Portfolio", use_container_width=True, key="append_ocr_btn") and st.session_state.active_package is not None:
                    updated_df = reg_engine.append_utility_bill_to_df(st.session_state.active_package, bill)
                    activate_portfolio(updated_df, f"{st.session_state.active_package.source_name} + OCR")
                    st.session_state.pending_bill = None
                    st.success("Bill appended and portfolio recalculated.")
                elif st.session_state.active_package is None:
                    st.info("Load a portfolio dataset first, then append the bill.")
            else:
                # Low confidence — show what was extracted with manual override form
                st.warning("⚠ Low confidence extraction. Review and correct fields below before appending.")
                with st.form("manual_bill_entry"):
                    st.markdown("**Manual Bill Entry** — fill in what you can read from the bill:")
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        m_energy = st.number_input("Energy (kWh)", min_value=0.0, value=float(bill.get("energy_kwh") or 0.0), step=100.0)
                        m_diesel = st.number_input("Diesel (Litres)", min_value=0.0, value=float(bill.get("diesel_litres") or 0.0), step=10.0)
                        m_water  = st.number_input("Water (kL)", min_value=0.0, value=float(bill.get("water_withdrawal_kl") or 0.0), step=1.0)
                    with m_col2:
                        m_cost   = st.number_input("Utility Cost (INR)", min_value=0.0, value=float(bill.get("utility_cost_inr") or 0.0), step=1000.0)
                        m_month  = st.text_input("Bill Month (YYYY-MM)", value=str(bill.get("bill_month") or ""))
                        m_asset  = st.text_input("Asset ID", value=str(bill.get("asset_id") or ""))
                    
                    submitted = st.form_submit_button("✅ Confirm & Append to Portfolio")
                    
                    if submitted and st.session_state.active_package is not None:
                        manual_bill = {
                            "energy_kwh": m_energy if m_energy > 0 else None,
                            "diesel_litres": m_diesel if m_diesel > 0 else None,
                            "water_withdrawal_kl": m_water if m_water > 0 else None,
                            "utility_cost_inr": m_cost if m_cost > 0 else None,
                            "bill_month": m_month if m_month.strip() else None,
                            "asset_id": m_asset.strip() if m_asset.strip() else None,
                            "_confidence": 1.0,  # human-verified
                            "_ocr_engine": "manual",
                        }
                        updated_df = reg_engine.append_utility_bill_to_df(st.session_state.active_package, manual_bill)
                        activate_portfolio(updated_df, f"{st.session_state.active_package.source_name} + Manual-Bill")
                        st.session_state.pending_bill = None
                        st.success("Manual bill entry confirmed and appended.")
                    elif submitted:
                        st.info("Load a portfolio dataset first, then append the bill.")

            if st.button("✕ Clear bill result", key="clear_bill_btn", use_container_width=False):
                st.session_state.pending_bill = None

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        # 4 hint cards across one row
        st.markdown(
            """<div class="hint-grid">
    <div class="hint-card">
        <div class="hint-title">✦ Rankings</div>
        <div class="hint-copy">"List the 5 most sustainable buildings and explain why."</div>
    </div>
    <div class="hint-card">
        <div class="hint-title">✦ Root Cause</div>
        <div class="hint-copy">"Why are emissions high in this building? Run a SHAP root cause."</div>
    </div>
    <div class="hint-card">
        <div class="hint-title">✦ BRSR Audit</div>
        <div class="hint-copy">"Run a full SEBI BRSR Principle 6 compliance audit."</div>
    </div>
    <div class="hint-card">
        <div class="hint-title">✦ CapEx Strategy</div>
        <div class="hint-copy">"What CapEx investments will improve our ESG score the most?"</div>
    </div>
</div>""",
            unsafe_allow_html=True,
        )

        # Chat — only render the shell when there's something to show
        has_messages = len(st.session_state.messages) > 0
        if has_messages:
            st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
            for idx, message in enumerate(st.session_state.messages):
                render_message(message, idx)
            st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.active_package is not None:
        render_command_snapshot(st.session_state.active_package)

# What-if scenario simulator
if not st.session_state.control_room:
    with st.expander("⊙ What-If Scenario Simulator", expanded=False):
        wif_left, wif_right = st.columns([1, 1], gap="large")
        with wif_left:
            solar_kw = st.slider("Solar PV Addition (kW)", 0, 500, 50, 10, key="wif_solar")
        with wif_right:
            setpoint_delta = st.slider("HVAC Setpoint Reduction (°C)", 0.0, 4.0, 1.0, 0.5, key="wif_hvac")
            
        st.caption(f"Simulate: +{solar_kw} kW solar  |  HVAC setpoint -{setpoint_delta}°C  |  Est. annual Scope 2 reduction: ~{solar_kw * 0.727 * 2400 / 1000:.1f} tCO₂")
        
        if st.button("Run What-If Simulation", use_container_width=True, key="wif_run"):
            queue_prompt(
                f"Run a what-if scenario: we install {solar_kw}kW of solar PV "
                f"and reduce HVAC setpoint by {setpoint_delta} degrees C. "
                f"Calculate estimated impact on Scope 1, Scope 2, total carbon, and energy intensity. "
                f"Show numbers and percentage changes."
            )

# Chat input — always rendered last so it docks correctly
prompt = st.session_state.queued_prompt or st.chat_input("Issue a data query, mutation, simulation, or export command")

if prompt:
    st.session_state.queued_prompt = ""
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if st.session_state.active_package is None:
        st.session_state.messages.append({"role": "assistant", "content": "[OMNI-SYSTEM] Load a dataset first."})
    else:
        if "export" in prompt.lower() and "dossier" in prompt.lower():
            _pkg = st.session_state.active_package
            with st.spinner("Building compliance dossier…"):
                try:
                    _chat_pdf, _chat_md = _build_pdf_now(_pkg)
                    st.session_state.dossier_pdf = _chat_pdf
                    st.session_state.dossier_md = _chat_md
                    _sz = f"{len(_chat_pdf):,} bytes" if _chat_pdf else "generation failed"
                    st.session_state.messages.append({"role": "assistant", "content": f"[OMNI-SYSTEM] Dossier complete — {_sz}. Scroll down to the Download section."})
                except Exception as _chat_e:
                    st.session_state.messages.append({"role": "assistant", "content": f"[OMNI-SYSTEM] PDF build error: {_chat_e}"})
        else:
            assistant_message, updated_package, rolling = reg_engine.chat_agent(st.session_state.messages, st.session_state.active_package, st.session_state.rolling_context)
            st.session_state.active_package = updated_package
            st.session_state.rolling_context = rolling
            st.session_state.session_carbon_g += carbon_tracker.estimate_prompt_footprint(prompt, updated_package.summary["records"])
            st.session_state.messages.append(assistant_message)
    st.rerun()

package = st.session_state.active_package

# NOTE: It's generally best practice in Python to define functions at the module level.
# `_build_pdf_now` is defined inside this `if` block, which means if `package is None`, 
# the download buttons at the bottom will fail if they try to access it.
if package is not None:
    render_command_snapshot(package)
    render_control_header(package)

    # Metric strip
    cards = st.columns(6, gap="small")
    card_data = [
        ("Scope 1  ·  CEA/IPCC", f"{package.summary['scope1_total']:.1f} tCO₂e"),
        ("Scope 2  ·  CEA v20", f"{package.summary['scope2_total']:.1f} tCO₂e"),
        ("Scope 3  ·  BRSR P6", f"{package.summary['scope3_total']:.1f} tCO₂e"),
        ("Total Carbon", f"{package.summary['total_carbon']:.1f} tCO₂e"),
        ("Renewable  ·  BEE", f"{(package.summary['solar_total']/max(package.summary['energy_total'],1))*100:.1f}%"),
        ("Anomalies  ·  IF 4%", str(package.summary['anomaly_count'])),
    ]
    for col, (label, value) in zip(cards, card_data):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

    figures: dict[str, go.Figure] = {}
    donut_col, energy_col = st.columns(2, gap="large")
    
    with donut_col:
        scope_df = pd.DataFrame({"Scope": ["Scope 1 (CEA/IPCC)", "Scope 2 (CEA v20)", "Scope 3 (SEBI BRSR)"], "tCO2e": [package.summary["scope1_total"], package.summary["scope2_total"], package.summary["scope3_total"]]})
        donut = px.pie(scope_df, names="Scope", values="tCO2e", hole=0.62, color="Scope", color_discrete_map={"Scope 1 (CEA/IPCC)": "#f85149", "Scope 2 (CEA v20)": "#58a6ff", "Scope 3 (SEBI BRSR)": "#00ff7f"})
        donut.update_layout(**fig_layout("GHG Scope Breakdown  ·  CEA v20 + IPCC AR4 + SEBI BRSR"))
        figures["Emissions Breakdown"] = donut
        render_plot(donut, key="emissions_breakdown_main")
        
    with energy_col:
        energy_fig = ai_engine.build_energy_figure(package)
        energy_fig.update_layout(**fig_layout("Monthly Energy Consumption  ·  Grid vs Solar  (CEA v20: 0.727 kg CO₂/kWh)"))
        figures["Monthly Energy"] = energy_fig
        render_plot(energy_fig, key="monthly_energy_main")

    ops_tab, assets_tab, anomalies_tab, mapping_tab = st.tabs([
        "[OMNI-SYSTEM] Control Room",
        "[DATA-QUERY] Asset Explorer",
        "[XAI-ALERT] Anomaly Console",
        "[CITED-RESEARCH] Data Mapping",
    ])

    with ops_tab:
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            # Reuse cached forecast — never recompute on every render
            if st.session_state.cached_forecast_df is None or st.session_state.get("_forecast_sig") != package.source_name:
                st.session_state.cached_forecast_df = ai_engine.forecast_portfolio(package.df, months=12)
                st.session_state["_forecast_sig"] = package.source_name
            forecast_df = st.session_state.cached_forecast_df
            forecast_fig = ai_engine.build_forecast_figure(forecast_df)
            figures["12-Month Forecast"] = forecast_fig
            render_plot(forecast_fig, key="twelve_month_forecast_main")
            
        with right:
            _recs = package.summary['records']
            _assets = package.summary['assets']
            _brsr = package.summary['brsr_principle']
            st.markdown(
                f"**System State**\n\n"
                f"- Records: `{_recs}` | Assets: `{_assets}`\n"
                f"- Framework: `SEBI BRSR Principle {_brsr}`\n"
                f"- Grid: `{DeterministicMath.CEA_FACTOR:.3f} kg CO₂/kWh  (CEA v20)`\n"
                f"- Anomaly: `Isolation Forest  contamination=4%  (SRD v3.0)`\n"
                f"- AI: `Precision >0.85  Recall >0.80  MAPE <15%`"
            )
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            if st.session_state.dossier_pdf:
                st.download_button("⬇ Download PDF", st.session_state.dossier_pdf, "cuesg_report.pdf", "application/pdf", use_container_width=True, key="ops_download_pdf")
                st.caption(f"PDF ready — {len(st.session_state.dossier_pdf):,} bytes")
            else:
                st.info("Use ⬡ Generate Compliance Report above to build the PDF.", icon="ℹ️")
            if st.session_state.dossier_md:
                st.download_button("⬇ Download Markdown", st.session_state.dossier_md, "cuesg_report.md", "text/markdown", use_container_width=True, key="ops_download_md")

    with assets_tab:
        selected_asset = st.selectbox("Asset", sorted(package.df["asset_id"].unique()))
        asset_df = package.df[package.df["asset_id"] == selected_asset].sort_values("month").reset_index(drop=True)
        asset_fig = go.Figure()
        asset_fig.add_scatter(x=asset_df["month"], y=asset_df["energy_kwh"], mode="lines+markers", name="Energy", line=dict(color="#58a6ff", width=2.5))
        asset_fig.add_scatter(x=asset_df["month"], y=asset_df["solar_kwh"], mode="lines+markers", name="Solar", line=dict(color="#00ff7f", width=2))
        asset_fig.update_layout(**fig_layout(f"{selected_asset} Trend"))
        render_plot(asset_fig, key=f"asset_trend_{selected_asset}")
        st.dataframe(asset_df.tail(12).reset_index(drop=True), use_container_width=True, hide_index=True)

    with anomalies_tab:
        if not package.anomaly_records.empty:
            sev_fig = px.scatter(package.anomaly_records.head(40), x="month", y="Anomaly_Score", color="Anomaly_Signature", size="energy_kwh", hover_data=["asset_id", "diesel_litres", "water_withdrawal_kl"])
            sev_fig.update_layout(**fig_layout("Anomaly Signature Monitor"))
            figures["Anomaly Signature Monitor"] = sev_fig
            render_plot(sev_fig, key="anomaly_signature_monitor")
            shap_fig = ai_engine.build_shap_figure(package)
            if shap_fig is not None:
                figures["SHAP Root-Cause Analysis"] = shap_fig
                shap_fig.update_layout(height=480)
                render_plot(shap_fig, key="anomaly_console_shap")
        st.dataframe(package.anomaly_records.head(20).reset_index(drop=True), use_container_width=True, hide_index=True)

    with mapping_tab:
        mapping_df = pd.DataFrame([{"Canonical": k, "Detected Source": v} for k, v in package.detected_columns.items()]).reset_index(drop=True)
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        st.markdown("#### Regulatory Reference — Source of Truth")
        _ref = [
            ("Grid Emission Factor","0.727 kg CO₂/kWh","CEA CO₂ Baseline Ver 20.0 (Dec 2024)"),
            ("High-Speed Diesel (HSD)","2.68 kg CO₂/litre","IPCC 2006 AR4 — BEE/MoEFCC"),
            ("PNG / Natural Gas","2.02 kg CO₂/m³","GAIL / IPCC 2006"),
            ("R-410A GWP","2,088","India Cooling Action Plan (ICAP) / IPCC AR4"),
            ("R-134a GWP","1,430","ICAP / IPCC AR4"),
            ("Water — Bengaluru","0.80–1.00 kL/m²/yr","BEE Benchmarking Grade-A Offices"),
            ("Water — Mumbai","1.25–1.50 kL/m²/yr","BEE Benchmarking Grade-A Offices"),
            ("Indoor Design Temp","24°C ± 1°C","ISHRAE / BEE ECSBC 2024"),
            ("Chiller Water-Cooled","0.65–0.75 kW/TR","Industrial HVAC Audit Standard India"),
            ("EUI Baseline","140–180 kWh/m²/yr","BEE Star Rating Baseline Commercial"),
            ("Anomaly Precision","> 0.85","GreenLens SRD v3.0 UC-2"),
            ("Anomaly Recall","> 0.80","GreenLens SRD v3.0 UC-2"),
            ("Forecast MAPE","< 15%","GreenLens SRD v3.0 UC-3"),
        ]
        st.dataframe(pd.DataFrame(_ref, columns=["Parameter","Value / Benchmark","Official Source"]),
                     use_container_width=True, hide_index=True)

    benchmark_shell_left, benchmark_shell_right = st.columns([1.0, 1.0], gap="large")
    with benchmark_shell_left:
        benchmark_df = (
            package.df.groupby("Climate_Zone", as_index=False)
            .agg(Avg_Energy_Intensity=("Energy_Intensity_kWh_per_sqm", "mean"), Avg_Water_Annualized=("Annualized_Water_Intensity", "mean"), Assets=("asset_id", "nunique"))
            .reset_index(drop=True)
        )
        benchmark_fig = go.Figure()
        benchmark_fig.add_bar(x=benchmark_df["Climate_Zone"], y=benchmark_df["Avg_Energy_Intensity"], name="Energy Intensity (kWh/m²)", marker_color="#58a6ff")
        # BEE EUI benchmark line
        benchmark_fig.add_scatter(x=benchmark_df["Climate_Zone"], y=[160]*len(benchmark_df), mode="lines", name="BEE EUI Baseline 160 kWh/m² (BEE Star Rating)", line=dict(color="#f85149", dash="dot", width=2))
        benchmark_fig.update_layout(**fig_layout("Energy Intensity vs BEE Baseline  ·  Source: BEE Star Rating (140–180 kWh/m²/yr)"), yaxis=dict(title="Energy Intensity (kWh/m²)"))
        figures["Climate Benchmark Monitor"] = benchmark_fig
        render_plot(benchmark_fig, key="climate_benchmark_monitor_main")
        
    with benchmark_shell_right:
        bee_fig = px.bar(
            package.asset_rankings.head(20).reset_index(drop=True),
            x="asset_id", y="Energy_Intensity_kWh_per_sqm", color="BEE_Rating", title="BEE Star Rating · Energy Intensity per Asset  |  Source: BEE ECSBC 2024",
            color_discrete_sequence=["#00ff7f", "#58a6ff", "#f78166", "#d29922", "#8b949e"],
        )
        bee_fig.update_layout(**fig_layout("BEE Rating · Energy Intensity (kWh/m²)"))
        render_plot(bee_fig, key="bee_leaderboard_main")

    # Use cached forecast — only compute once per source file
    if st.session_state.cached_forecast_df is None or st.session_state.get("_forecast_sig") != package.source_name:
        with st.spinner("Computing 12-month forecast..."):
            st.session_state.cached_forecast_df = ai_engine.forecast_portfolio(package.df, months=12)
            st.session_state["_forecast_sig"] = package.source_name
            
    forecast_df = st.session_state.cached_forecast_df
    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    render_detailed_dossier_workspace(package, forecast_df, figures)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── DOWNLOAD SECTION ──────────────────────────────────────────────────────
    def _build_pdf_now(pkg) -> tuple[bytes, str]:
        """Build and return (pdf_bytes, md_str) synchronously."""
        if st.session_state.cached_forecast_df is not None and st.session_state.get("_forecast_sig") == pkg.source_name:
            _fc = st.session_state.cached_forecast_df
        else:
            _fc = ai_engine.forecast_portfolio(pkg.df, months=12)
            st.session_state.cached_forecast_df = _fc
            st.session_state["_forecast_sig"] = pkg.source_name
            
        _s = pkg.summary
        _sd = pd.DataFrame({"Scope": ["Scope 1", "Scope 2", "Scope 3"],
                             "tCO2e": [_s["scope1_total"], _s["scope2_total"], _s["scope3_total"]]})
        _sf = ai_engine.build_shap_figure(pkg)
        _figs = {
            "Emissions Breakdown": px.pie(_sd, names="Scope", values="tCO2e", hole=0.6,
                color_discrete_map={"Scope 1": "#f85149", "Scope 2": "#58a6ff", "Scope 3": "#00ff7f"}),
            "12-Month Forecast": ai_engine.build_forecast_figure(_fc),
            "SHAP Root-Cause Analysis": _sf if _sf is not None else ai_engine.build_energy_figure(pkg),
            "Monthly Energy": ai_engine.build_energy_figure(pkg),
        }
        _md = reg_engine.build_markdown_report(pkg, _fc, st.session_state.messages, st.session_state.rolling_context)
        _pdf = reg_engine.build_pdf_report(_md, _figs, st.session_state.messages, st.session_state.rolling_context, pkg, forecast_df=_fc)
        return (_pdf if isinstance(_pdf, bytes) and len(_pdf) > 200 else b""), _md

    if st.session_state.get("_trigger_pdf_build") and st.session_state.active_package is not None:
        with st.spinner("Building compliance dossier…"):
            try:
                _ab_pdf, _ab_md = _build_pdf_now(st.session_state.active_package)
                st.session_state.dossier_pdf = _ab_pdf
                st.session_state.dossier_md = _ab_md
                st.session_state["_trigger_pdf_build"] = False
                if _ab_pdf:
                    st.success(f"✅ Dossier ready — {len(_ab_pdf):,} bytes. Download below.")
                else:
                    st.error("PDF build returned empty. Try again or check greenlens_pdf_engine.py.")
            except Exception as _e:
                st.error(f"PDF build failed: {_e}")
                st.session_state["_trigger_pdf_build"] = False

st.markdown(
    """<div style="background:rgba(0,255,127,0.04);border:1px solid rgba(0,255,127,0.15);border-radius:12px;padding:16px 20px;margin-bottom:12px;">
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.65rem;color:#00ff7f;letter-spacing:0.18em;text-transform:uppercase;font-weight:700;margin-bottom:6px;">
        ✦ Download Report
    </div>
    <div style="font-family:'Manrope',sans-serif;font-size:0.8rem;color:#4d5a6a;">
        Generate and instantly download your full report in one click.
    </div>
</div>""",
    unsafe_allow_html=True,
)

dl_left, dl_mid, dl_right = st.columns(3, gap="large")

# ---------------- PDF DOWNLOAD ---------------- #
with dl_left:
    _pkg_dl = st.session_state.get("active_package", None)

    if _pkg_dl is not None:
        try:
            # Generate instantly (no rerun, no state dependency)
            pdf_bytes, md_text = _build_pdf_now(_pkg_dl)

            if pdf_bytes and len(pdf_bytes) > 200:
                st.download_button(
                    label=f"⬇ Download PDF ({len(pdf_bytes)//1024} KB)",
                    data=pdf_bytes,
                    file_name="report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.warning("PDF generation failed.")

            # Store optional (for reuse)
            st.session_state.dossier_md = md_text or ""

        except Exception as e:
            st.error(f"PDF error: {e}")

    else:
        st.info("Load a dataset first to enable download.")

with dl_mid:
    if st.session_state.get("dossier_md"):
        st.download_button(
            "⬇ Download Markdown",
            st.session_state.dossier_md,
            "report.md",
            "text/markdown",
            use_container_width=True,
        )

with dl_right:
    if st.session_state.get("dossier_md"):
        st.download_button(
            "⬇ Download Text",
            st.session_state.dossier_md,
            "report.txt",
            "text/plain",
            use_container_width=True,
        )
