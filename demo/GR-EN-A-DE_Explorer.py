# GR-EN-A-DE Explorer — Interactive Streamlit Application
# (modified to stream uploads to temp files)
from __future__ import annotations

import html as _html
import io
import json
import os
import pickle
import warnings
import shutil
import tempfile
import gzip
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from urllib.parse import urlparse

try:
    import joblib
except Exception:
    joblib = None

warnings.filterwarnings("ignore")

SPRING_LAYOUT_K_FACTOR = 2.5
SPRING_LAYOUT_ITERATIONS = 60
OLLAMA_TIMEOUT_SECONDS = 120
_ALLOWED_OLLAMA_SCHEMES = ("http", "https")
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 20


def _safe_local_path(raw: str) -> Path:
    if "\x00" in raw:
        raise ValueError("Path contains null bytes.")
    return Path(raw).resolve()


def _validate_ollama_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_OLLAMA_SCHEMES:
        raise ValueError(
            f"Disallowed URL scheme '{parsed.scheme}'. Only http and https are permitted."
        )
    return url


def save_uploaded_to_temp(uploaded_file, suffix: Optional[str] = None) -> Path:
    suffix = suffix or (Path(uploaded_file.name).suffix or "")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp)
        tmp_path = Path(tmp.name)
    return tmp_path


def download_to_temp(url: str, progress_text: str = "Downloading…") -> Path:
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    ext = Path(urlparse(url).path).suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    downloaded = 0
    progress = st.progress(0, text=progress_text)
    try:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                break
            tmp.write(chunk)
            downloaded += len(chunk)
            if total:
                progress.progress(min(downloaded / total, 1.0))
    finally:
        tmp.close()
        progress.empty()
    return Path(tmp.name)


st.set_page_config(
    page_title="GR-EN-A-DE Explorer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_COLORS = {
    "orange": "#f97316",
    "amber":  "#f59e0b",
    "blue":   "#1e40af",
    "indigo": "#1d4ed8",
    "sky":    "#60a5fa",
    "teal":   "#4db8c8",
    "green":  "#34d399",
    "pink":   "#e11d48",
    "muted":  "#626567",
}

PALETTE = [
    THEME_COLORS["orange"],
    THEME_COLORS["blue"],
    THEME_COLORS["sky"],
    THEME_COLORS["teal"],
    THEME_COLORS["indigo"],
    THEME_COLORS["green"],
    THEME_COLORS["pink"],
    "#c0392b", "#27ae60", "#d4ac0d", "#6c3483", "#a93226",
    "#1a7a4a", "#1f618d", "#5b2c6f", "#7e5109", "#0e6655",
    "#6e2c00", "#626567", "#935116",
]


def _cat_color(idx: int) -> str:
    return PALETTE[int(idx) % len(PALETTE)]


def _map_norm_to_palette(norm: float) -> str:
    idx = int(norm * (len(PALETTE) - 1) + 0.5)
    return PALETTE[max(0, min(idx, len(PALETTE) - 1))]


def _ss() -> Any:
    return st.session_state


def _init_state():
    defaults = {
        "raw_adj": None,
        "node_df": None,
        "n": 0,
        "cols": [],
        "cat_cols": [],
        "node_attrs": {},
        "imp_data": {},
        "selected_node": None,
        "selected_edge": None,
        "xain_results": [],
        "dark_mode": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _t() -> dict:
    dark = st.session_state.get("dark_mode", True)
    if dark:
        return dict(
            plotly_bg="#0b0f1a",
            plotly_panel="#111827",
            plotly_grid="#1f2937",
            plotly_font="#e5e7eb",
            plotly_title="white",
            legend_bg="rgba(255,255,255,0.96)",
            legend_border="#aaaaaa",
            legend_font="#1e293b",
            bar_marker="#34d399",
            bar_mline="rgba(255,255,255,0.3)",
        )
    return dict(
        plotly_bg="#f7f4f0",
        plotly_panel="#ffffff",
        plotly_grid="#e2e8f0",
        plotly_font="#1e293b",
        plotly_title="#1e293b",
        legend_bg="#ffffff",
        legend_border="#e2e8f0",
        legend_font="#1e293b",
        bar_marker="#059669",
        bar_mline="rgba(0,0,0,0.08)",
    )








def _inject_theme_css(dark: bool) -> None:
    if dark:
        theme_vars = """
:root {
    --text-heading:        #60a5fa;
    --text-secondary:      #d1d5db;
    --text-subheading:     #f59e0b;
    --text-accent-green:   #34d399;
    --text-link:           #93c5fd;
    --text-muted:          #9ca3af;
    --text-primary:        #f9fafb;
    --code-color:          #7dd3fc;
    --border-footer:       #1f2937;
    --text-footer:         #6b7280;
    --td-border:           #1f2937;
    --xain-th-bg:          #1e3a5f;
    --xain-th-border:      #374151;
    --xain-row-even:       #0f172a;
    --xain-row-odd:        #111827;
    --bg-card:             rgba(17,24,39,0.75);
    --bg-metric:           rgba(31,41,55,0.8);
    --bg-table-th:         #1f2937;
    --bg-table-hover:      #1f2937;
    --bg-tabs:             rgba(17,24,39,0.8);
    --border-card:         rgba(96,165,250,0.12);
    --border-metric:       rgba(96,165,250,0.15);
    --border-tab:          rgba(96,165,250,0.12);
    --tab-text:            #d1d5db;
    --section-title:       #f9fafb;
    --section-badge-border:rgba(249,115,22,0.5);
    --section-badge-color: #f97316;
    --section-divider:     linear-gradient(90deg,#f97316,#e11d48);
    --section-desc:        #d1d5db;
    --feature-card-bg:     rgba(17,24,39,0.75);
    --feature-card-border: rgba(96,165,250,0.14);
    --feature-name:        #f9fafb;
    --feature-desc:        #d1d5db;
}"""
        app_bg = """
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #1a1040 0%, #0b0e1a 50%, #060a14 100%);
    color: #e5e7eb; min-height: 100vh;
}
.stApp::before {
    content: ''; position: fixed; inset: 0;
    background-image:
        radial-gradient(1px 1px at 15% 25%, rgba(255,255,255,0.55) 0%, transparent 100%),
        radial-gradient(1px 1px at 72% 12%, rgba(255,255,255,0.45) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 68%, rgba(255,255,255,0.40) 0%, transparent 100%),
        radial-gradient(1px 1px at 88% 55%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 80%, rgba(255,255,255,0.30) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 40%, rgba(255,255,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at  5% 90%, rgba(255,255,255,0.20) 0%, transparent 100%),
        radial-gradient(1px 1px at 93% 78%, rgba(255,255,255,0.20) 0%, transparent 100%);
    pointer-events: none; z-index: 0;
}
section[data-testid="stSidebar"] {
    background: rgba(10,14,30,0.95) !important;
    border-right: 1px solid rgba(96,165,250,0.15);
}
/* ── Dark: file uploader dropzone ───────────────────────────────── */
div[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1.5px dashed rgba(96, 165, 250, 0.30) !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p,
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] button { color: #e5e7eb !important; }
/* ── Dark: uploaded file chip ────────────────────────────────────── */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    background-color: rgba(30, 41, 59, 0.90) !important;
    color: #e5e7eb !important;
    border-color: rgba(96, 165, 250, 0.20) !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg {
    color: #94a3b8 !important;
    background: transparent !important;
    box-shadow: none !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover {
    color: #f87171 !important;
    background: rgba(239,68,68,0.15) !important;
}
/* ── Dark: number input ─────────────────────────────────────────── */
div[data-testid="stNumberInput"] input {
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: #f9fafb !important;
    border: 1px solid rgba(96, 165, 250, 0.25) !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] button {
    background-color: rgba(30, 41, 59, 0.90) !important;
    color: #e5e7eb !important;
    border-color: rgba(96, 165, 250, 0.20) !important;
    box-shadow: none !important;
}
div[data-testid="stNumberInput"] button:hover {
    background-color: rgba(96, 165, 250, 0.15) !important;
}
/* ── Dark: text input ───────────────────────────────────────────── */
div[data-testid="stTextInput"] input {
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: #f9fafb !important;
    border: 1px solid rgba(96, 165, 250, 0.25) !important;
    border-radius: 8px !important;
}
/* ── Dark: selectbox control box ────────────────────────────────── */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(96, 165, 250, 0.25) !important;
    border-radius: 8px !important;
}
/* ── Dark: selectbox / multiselect dropdown listbox (portal) ─────── */
[data-baseweb="popover"],
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background-color: #0f172a !important;
    border: 1px solid rgba(96, 165, 250, 0.25) !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: rgba(96, 165, 250, 0.15) !important;
    color: #60a5fa !important;
}
[data-baseweb="popover"] [data-highlighted="true"] {
    background-color: rgba(96, 165, 250, 0.20) !important;
    color: #93c5fd !important;
}
[data-baseweb="popover"] ::-webkit-scrollbar { width: 6px; }
[data-baseweb="popover"] ::-webkit-scrollbar-track { background: #0f172a; }
[data-baseweb="popover"] ::-webkit-scrollbar-thumb {
    background: rgba(96,165,250,0.30); border-radius: 3px;
}"""
    else:
        theme_vars = """
:root {
    --text-heading:        #1e40af;
    --text-secondary:      #64748b;
    --text-subheading:     #ea580c;
    --text-accent-green:   #059669;
    --text-link:           #3b82f6;
    --text-muted:          #94a3b8;
    --text-primary:        #1e293b;
    --code-color:          #2563eb;
    --border-footer:       #e2e8f0;
    --text-footer:         #94a3b8;
    --td-border:           #e2e8f0;
    --xain-th-bg:          #e0f2fe;
    --xain-th-border:      #bae6fd;
    --xain-row-even:       #f0f9ff;
    --xain-row-odd:        #ffffff;
    --bg-card:             #ffffff;
    --bg-metric:           #f8fafc;
    --bg-table-th:         #f1f5f9;
    --bg-table-hover:      #f8fafc;
    --bg-tabs:             #f1f5f9;
    --border-card:         #e2e8f0;
    --border-metric:       #e2e8f0;
    --border-tab:          #e2e8f0;
    --tab-text:            #64748b;
    --section-title:       #1e293b;
    --section-badge-border:rgba(14,116,144,0.5);
    --section-badge-color: #0e7490;
    --section-divider:     linear-gradient(90deg,#0284c7,#0e7490);
    --section-desc:        #64748b;
    --feature-card-bg:     #ffffff;
    --feature-card-border: #e2e8f0;
    --feature-name:        #1e293b;
    --feature-desc:        #64748b;
}"""
        app_bg = """
.stApp {
    background: linear-gradient(160deg, #eef2ff 0%, #f7f4f0 45%, #f0f7ff 100%);
    color: #1e293b; min-height: 100vh;
}
.stApp::before { display: none; }
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
/* ── Light: file uploader dropzone ──────────────────────────────── */
div[data-testid="stFileUploaderDropzone"] {
    background-color: #f8fafc !important;
    border: 1.5px dashed #94a3b8 !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p,
div[data-testid="stFileUploaderDropzone"] small,
div[data-testid="stFileUploaderDropzone"] button { color: #1e293b !important; }
/* ── Light: uploaded file chip ───────────────────────────────────── */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button {
    color: #64748b !important;
    background: transparent !important;
    box-shadow: none !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button:hover {
    color: #ef4444 !important;
    background: rgba(239,68,68,0.08) !important;
}
/* ── Light: number input ─────────────────────────────────────────── */
div[data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] button {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
    box-shadow: none !important;
}
/* ── Light: selectbox / multiselect dropdown listbox (portal) ────── */
[data-baseweb="popover"],
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li {
    background-color: #ffffff !important;
    color: #1e293b !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #eff6ff !important;
    color: #1e40af !important;
}
[data-baseweb="popover"] [data-highlighted="true"] {
    background-color: #dbeafe !important;
    color: #1e40af !important;
}"""

    base_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html, body { font-family: 'Inter','Segoe UI',sans-serif; }

/* ── Widget labels ────────────────────────────────────────────────── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stRadio"] label,
div[data-testid="stToggle"] label { color: var(--text-primary) !important; }
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] p { color: var(--text-primary) !important; }
div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div { color: var(--text-primary) !important; }
ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[data-testid="stMultiSelectDropdown"] li { color: var(--text-primary) !important; }
div[data-testid="stMultiSelect"] span,
div[data-testid="stMultiSelect"] div[data-baseweb="tag"] span,
div[data-testid="stMultiSelect"] input,
div[data-testid="stMultiSelect"] input::placeholder { color: var(--text-primary) !important; }
div[data-testid="stNumberInput"] input { color: var(--text-primary) !important; }
div[data-testid="stTextInput"] input { color: var(--text-primary) !important; }
div[data-testid="stCheckbox"] p,
div[data-testid="stCheckbox"] span { color: var(--text-primary) !important; }
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span { color: var(--text-primary) !important; }
div[data-testid="stMarkdownContainer"] p { color: var(--text-primary); }

/* ── Sidebar labels & text ────────────────────────────────────────── */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label,
section[data-testid="stSidebar"] div[data-testid="stSlider"] label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] label,
section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] label,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] label,
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label,
section[data-testid="stSidebar"] div[data-testid="stToggle"] label { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] div[data-testid="stSlider"] p,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] div { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3 { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] { backdrop-filter: blur(10px); }

/* ── Main action buttons ──────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #f97316 0%, #e11d48 100%) !important;
    color: white !important; font-weight: 700; border-radius: 10px;
    border: none; padding: 10px 16px;
    box-shadow: 0 4px 14px rgba(249,115,22,0.4);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(249,115,22,0.6);
}

/* ── Tabs ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-tabs); border-radius: 50px; padding: 4px 6px; gap: 4px;
    border: 1px solid var(--border-tab);
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--tab-text);
    border-radius: 50px; font-weight: 600; font-size: 0.85rem;
    padding: 8px 22px; transition: background 0.2s, color 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
    color: white !important; box-shadow: 0 2px 12px rgba(30,64,175,0.45);
}

/* ── Cards ────────────────────────────────────────── */
.card {
    background: var(--bg-card); border: 1px solid var(--border-card);
    border-radius: 14px; padding: 18px 22px; margin-bottom: 12px;
    backdrop-filter: blur(6px);
}

/* ── Hero ─────────────────────────────────────────── */
.hero-section { text-align: center; padding: 100px 20px 60px; }
.hero-title {
    font-size: clamp(2.4rem,6vw,4.2rem); font-weight: 900;
    line-height: 1.1; margin-bottom: 20px; letter-spacing: -0.02em;
}
.hero-title .word-orange { color: #f97316; }
.hero-title .word-blue   { color: #4db8c8; }
.hero-subtitle {
    color: var(--text-secondary); font-size: 1.05rem;
    max-width: 560px; margin: 0 auto 32px; line-height: 1.7; text-align: center;
}
.hero-cta {
    display: inline-block;
    background: linear-gradient(135deg, #f97316 0%, #e11d48 100%);
    color: white !important; font-weight: 700; font-size: 0.95rem;
    padding: 14px 38px; border-radius: 50px; text-decoration: none;
    letter-spacing: 0.04em; box-shadow: 0 4px 24px rgba(249,115,22,0.40);
    transition: transform 0.2s, box-shadow 0.2s;
}
.hero-cta:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(249,115,22,0.55); }

/* ── Section badge ────────────────────────────────── */
.section-badge {
    display: inline-block; border: 1px solid var(--section-badge-border);
    border-radius: 50px; color: var(--section-badge-color);
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 5px 14px; margin-bottom: 18px;
}
.section-title { font-size: 2rem; font-weight: 800; color: var(--section-title); text-align: center; margin-bottom: 8px; }
.section-divider { width: 50px; height: 3px; background: var(--section-divider); margin: 12px auto 20px; border-radius: 2px; }
.section-desc { color: var(--section-desc); text-align: center; font-size: 0.95rem; max-width: 520px; margin: 0 auto 36px; line-height: 1.6; }

/* ── Feature cards ────────────────────────────────── */
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 18px; max-width: 900px; margin: 0 auto; }
.feature-card {
    background: var(--feature-card-bg); border: 1px solid var(--feature-card-border);
    border-radius: 14px; padding: 22px 18px; text-align: center;
    backdrop-filter: blur(6px); transition: border-color 0.2s, transform 0.2s;
}
.feature-card:hover { border-color: rgba(249,115,22,0.4); transform: translateY(-3px); }
.feature-icon { font-size: 2rem; margin-bottom: 10px; }
.feature-name { font-weight: 700; color: var(--feature-name); font-size: 0.95rem; margin-bottom: 6px; }
.feature-desc { color: var(--feature-desc); font-size: 0.8rem; line-height: 1.5; text-align: center; }

/* ── Metric tiles ─────────────────────────────────── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.metric {
    background: var(--bg-metric); border: 1px solid var(--border-metric);
    border-radius: 10px; padding: 10px 20px; text-align: center;
    min-width: 100px; backdrop-filter: blur(4px);
}
.metric-val { font-size: 1.6rem; font-weight: 700; color: var(--text-heading); }
.metric-lbl { font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: .05em; }

/* ── Node info table ─────────────────────────────── */
.node-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.node-table th { background: var(--bg-table-th); color: var(--text-subheading); padding: 7px 12px; text-align: left; }
.node-table td { padding: 6px 12px; border-top: 1px solid var(--border-card); color: var(--text-primary); }
.node-table tr:hover td { background: var(--bg-table-hover); }

/* ── Explanation table ────────────────────────────── */
.xain-table { border-collapse: collapse; width: 100%; font-size: 12px; font-family: monospace; }
.xain-table th { background: var(--xain-th-bg); padding: 9px 12px; border: 1px solid var(--xain-th-border); color: var(--text-subheading); }
.xain-table td { padding: 7px 10px; border: 1px solid var(--td-border); vertical-align: top; color: var(--text-primary); }
.xain-table tr:nth-child(even) td { background: var(--xain-row-even); }
.xain-table tr:nth-child(odd)  td { background: var(--xain-row-odd); }

/* ── Scrollable div / badges ──────────────────────── */
.scroll-x { overflow-x: auto; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; margin-right:4px; }
.badge-blue  { background:#1e40af; color:#bfdbfe; }
.badge-green { background:#065f46; color:#a7f3d0; }
.badge-amber { background:#78350f; color:#fde68a; }
.badge-red   { background:#7f1d1d; color:#fca5a5; }

.hero-wrapper { display: flex; flex-direction: column; align-items: center; }

/* ── Expander fixes ───────────────────────────────── */
section[data-testid="stExpander"] summary {
    display: flex !important; align-items: center !important;
    justify-content: space-between !important; white-space: normal !important;
    overflow: visible !important; background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(96,165,250,0.2); border-radius: 10px;
    padding: 10px 14px; color: #e5e7eb !important; font-weight: 600;
}
section[data-testid="stExpander"] summary * { position: relative !important; z-index: 1 !important; }
section[data-testid="stExpander"] { overflow: visible !important; }
"""
    st.markdown(f"<style>{theme_vars}{app_bg}{base_css}</style>", unsafe_allow_html=True)

























# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_adj(source) -> Optional[np.ndarray]:
    try:
        if isinstance(source, (bytes, bytearray)):
            raw = pickle.loads(source)
        elif isinstance(source, (str, Path)):
            p = Path(source)
            if p.suffix == ".gz":
                with gzip.open(p, "rb") as f:
                    raw = pickle.load(f)
            else:
                if joblib is not None and p.suffix in {".joblib", ".jbl"}:
                    raw = joblib.load(p)
                else:
                    with open(p, "rb") as f:
                        raw = pickle.load(f)
        else:
            source.seek(0)
            head = source.read(2)
            source.seek(0)
            if head == b"\x1f\x8b":
                with gzip.GzipFile(fileobj=source) as f:
                    raw = pickle.load(f)
            else:
                raw = pickle.load(source)
        arr = np.array(raw.numpy() if hasattr(raw, "numpy") else raw, dtype=np.float32)
        np.fill_diagonal(arr, 0)
        return arr
    except Exception as e:
        st.error(f"Failed to load adjacency matrix: {e}")
        return None


def _load_csv(source) -> Optional[pd.DataFrame]:
    try:
        if isinstance(source, (bytes, bytearray)):
            df = pd.read_csv(io.BytesIO(source), index_col=0).reset_index(drop=True)
        elif isinstance(source, (str, Path)):
            df = pd.read_csv(source, index_col=0).reset_index(drop=True)
        else:
            source.seek(0)
            df = pd.read_csv(source, index_col=0).reset_index(drop=True)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.strip().str.strip(".,;:").str.strip()
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None


def _store_data(adj: np.ndarray, df: pd.DataFrame):
    st.session_state.raw_adj = adj
    st.session_state.node_df = df
    st.session_state.n = adj.shape[0]
    st.session_state.cols = list(df.columns)
    st.session_state.cat_cols = [
        c for c in df.columns
        if df[c].dtype == object or df[c].nunique() <= MAX_UNIQUE_VALUES_FOR_CATEGORICAL
    ]
    st.session_state.node_attrs = df.to_dict("index")


def _data_loaded() -> bool:
    return st.session_state.raw_adj is not None and st.session_state.node_df is not None


# ─────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_top_nodes_by_degree(subset: set, max_nodes: int) -> list:
    adj = st.session_state.raw_adj
    deg = np.array((adj > 0).sum(axis=1))
    return sorted(subset, key=lambda n: -deg[n])[:max_nodes]


def _build_plotly_graph(
    node_list: list,
    edge_list: list,
    node_df: pd.DataFrame,
    colour_col: str,
    cat_cols: list,
    node_sizes: Optional[np.ndarray] = None,
    edge_weights: Optional[list] = None,
    edge_color_fixed: str = THEME_COLORS["sky"],
    title: str = "",
    show_ids: bool = True,
    highlight_node: Optional[int] = None,
    highlight_edges: Optional[list] = None,
) -> go.Figure:
    n_nodes = len(node_list)
    G_tmp = nx.Graph()
    G_tmp.add_nodes_from(node_list)
    G_tmp.add_edges_from(edge_list)
    if n_nodes <= 80:
        pos = nx.kamada_kawai_layout(G_tmp)
    else:
        pos = nx.spring_layout(
            G_tmp, seed=42,
            k=SPRING_LAYOUT_K_FACTOR / np.sqrt(max(n_nodes, 1)),
            iterations=SPRING_LAYOUT_ITERATIONS,
        )
    pos_x = {n: pos[n][0] for n in node_list}
    pos_y = {n: pos[n][1] for n in node_list}

    if colour_col != "(none)":
        uniq = sorted(node_df[colour_col].dropna().unique().tolist(), key=str)
        cat_idx = {v: i for i, v in enumerate(uniq)}
        node_colors = [_cat_color(cat_idx.get(node_df[colour_col].iloc[nd], 0)) for nd in node_list]
    else:
        deg_arr = np.array([G_tmp.degree(nd) for nd in node_list], dtype=float)
        d_min, d_max = deg_arr.min(), deg_arr.max()
        norm = (deg_arr - d_min) / (d_max - d_min + 1e-9)
        node_colors = [_map_norm_to_palette(0.1 + 0.80 * v) for v in norm]

    if node_sizes is None:
        deg_arr = np.array([G_tmp.degree(nd) for nd in node_list], dtype=float)
        d_min, d_max = deg_arr.min(), deg_arr.max()
        node_sizes = 20 + 45 * (deg_arr - d_min) / (d_max - d_min + 1e-9)

    border_colors = []
    border_widths = []
    for nd in node_list:
        if nd == highlight_node:
            border_colors.append(THEME_COLORS["amber"])
            border_widths.append(3)
        else:
            border_colors.append("rgba(255,255,255,0.6)")
            border_widths.append(0.8)

    def _hover(nd):
        row = node_df.iloc[nd]
        lines = [f"<b>Node {nd}</b>"]
        for col in node_df.columns[:8]:
            lines.append(f"{col}: {str(row[col])[:80]}")
        return "<br>".join(lines)

    hover_texts = [_hover(nd) for nd in node_list]
    edge_traces = []
    h_edges = set(map(tuple, highlight_edges)) if highlight_edges else set()

    if edge_weights is not None:
        ew_arr = np.array(edge_weights, dtype=float)
        ew_min, ew_max = ew_arr.min(), ew_arr.max()
        ew_norm = (ew_arr - ew_min) / (ew_max - ew_min + 1e-9)
        for i, (u, v) in enumerate(edge_list):
            if u not in pos_x or v not in pos_x:
                continue
            alpha = 0.4 + 0.55 * ew_norm[i]
            hex_c = _map_norm_to_palette(ew_norm[i])
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            color = f"rgba({r},{g},{b},{alpha:.2f})"
            width = 1 + 4 * ew_norm[i]
            is_hl = (u, v) in h_edges or (v, u) in h_edges
            edge_traces.append(go.Scatter(
                x=[pos_x[u], pos_x[v], None], y=[pos_y[u], pos_y[v], None],
                mode="lines",
                line=dict(width=width if not is_hl else width + 2,
                          color=THEME_COLORS["amber"] if is_hl else color),
                hoverinfo="none", showlegend=False,
            ))
    else:
        ex_reg, ey_reg, ex_hl, ey_hl = [], [], [], []
        for u, v in edge_list:
            if u not in pos_x or v not in pos_x:
                continue
            if (u, v) in h_edges or (v, u) in h_edges:
                ex_hl += [pos_x[u], pos_x[v], None]
                ey_hl += [pos_y[u], pos_y[v], None]
            else:
                ex_reg += [pos_x[u], pos_x[v], None]
                ey_reg += [pos_y[u], pos_y[v], None]
        if ex_reg:
            edge_traces.append(go.Scatter(
                x=ex_reg, y=ey_reg, mode="lines",
                line=dict(width=1.2, color=edge_color_fixed),
                opacity=0.55, hoverinfo="none", showlegend=False,
            ))
        if ex_hl:
            edge_traces.append(go.Scatter(
                x=ex_hl, y=ey_hl, mode="lines",
                line=dict(width=3, color=THEME_COLORS["amber"]),
                opacity=0.9, hoverinfo="none", showlegend=False,
            ))

    node_trace = go.Scatter(
        x=[pos_x[nd] for nd in node_list],
        y=[pos_y[nd] for nd in node_list],
        mode="markers",
        marker=dict(
            size=node_sizes, color=node_colors,
            line=dict(color=border_colors, width=border_widths), opacity=0.95,
        ),
        hovertext=hover_texts, hoverinfo="text",
        customdata=node_list, showlegend=False, name="nodes",
    )

    label_annotations = []
    if show_ids and n_nodes <= 150:
        for nd in node_list:
            label_annotations.append(dict(
                x=pos_x[nd], y=pos_y[nd], text=f"<b>{nd}</b>",
                showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.5,
                arrowcolor="rgba(200,200,200,0.7)", ax=0, ay=-32,
                font=dict(size=10, color="#ffffff", family="Inter, sans-serif"),
                bgcolor="rgba(20,24,48,0.88)", bordercolor="rgba(180,180,255,0.45)",
                borderwidth=1, borderpad=3, xref="x", yref="y",
                xanchor="center", yanchor="bottom",
            ))

    legend_traces = []
    if colour_col != "(none)":
        uniq = sorted(node_df[colour_col].dropna().unique().tolist(), key=str)
        cat_idx = {v: i for i, v in enumerate(uniq)}
        shown = sorted({node_df[colour_col].iloc[nd] for nd in node_list}, key=str)
        for v in shown:
            legend_traces.append(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=_cat_color(cat_idx[v])),
                name=str(v)[:40], showlegend=True,
            ))

    fig = go.Figure(
        data=edge_traces + [node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(color=_t()["plotly_title"], size=15)),
            paper_bgcolor=_t()["plotly_bg"], plot_bgcolor=_t()["plotly_bg"],
            font=dict(color=_t()["plotly_font"]),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                bgcolor=_t()["legend_bg"], bordercolor=_t()["legend_border"],
                borderwidth=1, font=dict(size=11, color=_t()["legend_font"]),
                title=dict(text=colour_col if colour_col != "(none)" else "",
                           font=dict(size=12, color=_t()["legend_font"])),
                x=1.01, xanchor="left",
            ),
            margin=dict(l=10, r=10, t=50, b=10), height=750,
            hovermode="closest", clickmode="event+select",
            annotations=label_annotations,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GraphXAIn helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_think(raw: str) -> str:
    if "<think>" in raw and "</think>" in raw:
        return raw[raw.rfind("</think>") + len("</think>"):].strip()
    return raw.strip()


def _parse_json(raw: str) -> dict:
    clean = _strip_think(raw)
    if "```" in clean:
        for part in clean.split("```"):
            part = part.strip().lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    try:
        return json.loads(clean)
    except Exception:
        return {"explanation": clean}


def _node_attr(nid: int, col: Optional[str]) -> str:
    df = st.session_state.node_df
    if col and df is not None and col in df.columns and 0 <= nid < len(df):
        return str(df[col].iloc[nid])
    return "N/A"


def _node_text(nid: int, text_cols: list) -> str:
    df = st.session_state.node_df
    if text_cols and df is not None and 0 <= nid < len(df):
        return str(df[text_cols[0]].iloc[nid])
    return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    dark_mode = st.toggle("🌙 Night mode", key="dark_mode")
    st.markdown(
        "<h2 style='color:var(--text-heading);margin-bottom:4px'>🌐 GR\u2011EN\u2011A\u2011DE Explorer</h2>"
        "<p style='color:var(--text-muted);font-size:12px;margin-top:0'>Adjacency Matrix Visual Analytics</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 📂 Load Data")

    adj_upload = st.file_uploader(
        "Adjacency `.pkl` / `.pkl.gz` / `.joblib`",
        type=["pkl", "gz", "joblib", "pickle"],
        help="Pickle (optionally gzipped) or joblib adjacency matrix file",
    )
    csv_upload = st.file_uploader("Dataset `.csv`", type=["csv"])

    if st.button("🚀 Load files", use_container_width=True):
        try:
            if not adj_upload or not csv_upload:
                st.warning("Please upload both files.")
            else:
                st.info("Saving uploads to temporary files…")
                tmp_adj = save_uploaded_to_temp(adj_upload, suffix=Path(adj_upload.name).suffix or ".pkl")
                tmp_csv = save_uploaded_to_temp(csv_upload, suffix=Path(csv_upload.name).suffix or ".csv")
                adj = _load_adj(str(tmp_adj))
                df = _load_csv(str(tmp_csv))
                if adj is not None and df is not None:
                    _store_data(adj, df)
                    st.success(f"✔ Matrix {adj.shape} · {len(df)} nodes")
        except Exception as e:
            st.error(f"Failed to load files: {e}")

    if _data_loaded():
        N = st.session_state.n
        cat_cols = st.session_state.cat_cols
        st.divider()
        st.markdown("### 📊 Quick Stats")
        st.markdown(
            f"<div class='metric-row'>"
            f"<div class='metric'><div class='metric-val'>{N}</div><div class='metric-lbl'>Nodes</div></div>"
            f"<div class='metric'><div class='metric-val'>{int(np.sum(st.session_state.raw_adj > 0) // 2)}</div><div class='metric-lbl'>Edges</div></div>"
            f"<div class='metric'><div class='metric-val'>{len(st.session_state.cols)}</div><div class='metric-lbl'>Features</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### 🎛 Global Graph Controls")
        g_colour_col = st.selectbox("Colour nodes by", ["(none)"] + cat_cols, key="g_colour")
        g_max_nodes = st.slider("Max nodes displayed", 10, min(N, 500), min(300, N), 10, key="g_max")
        g_show_ids = st.checkbox("Show node IDs (≤150)", value=True, key="g_show_ids")

_inject_theme_css(st.session_state.get("dark_mode", True))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

if not _data_loaded():
    st.markdown(
        """
<div class="hero-wrapper">
  <div class='hero-section'>
    <div class='hero-title' style='white-space:nowrap'>
      <span class='word-orange'>GR-EN-A-DE</span>&nbsp;
      <span style='color:var(--text-primary)'> Explorer</span>
    </div>
  </div>
  <div style='text-align:center;padding:20px 0 10px'>
    <div class='section-badge'>TOOL SECTIONS</div>
    <div class='section-title'>What You Can Explore</div>
    <div class='section-divider'></div>
    <p class='section-desc'>
      Load your <code style='color:var(--code-color)'>.pkl</code> adjacency matrix and
      <code style='color:var(--code-color)'>.csv</code> node features from the sidebar to unlock all sections.
    </p>
  </div>
</div>
<div class='feature-grid'>
  <div class='feature-card'><div class='feature-icon'>🔭</div>
    <div class='feature-name'>Raw Graph</div>
    <div class='feature-desc'>Interactive Plotly network coloured by node attributes. Click any node to inspect it.</div></div>
  <div class='feature-card'><div class='feature-icon'>🔍</div>
    <div class='feature-name'>Node Inspector</div>
    <div class='feature-desc'>Feature table, degree, and neighbourhood sub-graph for any selected node.</div></div>
  <div class='feature-card'><div class='feature-icon'>⚡</div>
    <div class='feature-name'>Edge Importance</div>
    <div class='feature-desc'>Overlay GNN importance scores on the graph and explore top-ranked edges.</div></div>
  <div class='feature-card'><div class='feature-icon'>🧠</div>
    <div class='feature-name'>LLM Explanations</div>
    <div class='feature-desc'>Generate human-readable explanations via Ollama for the most important edges.</div></div>
</div>
""", unsafe_allow_html=True)
    st.stop()

RAW_ADJ: np.ndarray = st.session_state.raw_adj
NODE_DF: pd.DataFrame = st.session_state.node_df
N: int = st.session_state.n
COLS: list = st.session_state.cols
CAT_COLS: list = st.session_state.cat_cols
NODE_ATTRS: dict = st.session_state.node_attrs
colour_col: str = st.session_state.get("g_colour", "(none)")
max_nodes: int = st.session_state.get("g_max", min(300, N))
show_ids: bool = st.session_state.get("g_show_ids", True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔭 Raw Graph", "🔍 Node Inspector", "⚡ Edge Importance", "🧠 LLM Explanations",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Raw Graph
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "<h3 style='color:var(--text-heading)'>Raw Graph Exploration</h3>"
        "<p style='color:var(--text-secondary);font-size:13px'>"
        "Colour nodes by category · filter by label · click a node to inspect it in the Node Inspector tab."
        "</p>", unsafe_allow_html=True,
    )
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        t1_colour = st.selectbox("Colour by", ["(none)"] + CAT_COLS, key="t1_colour")
    with col_ctrl2:
        t1_max = st.slider("Max nodes", 10, min(N, 500), min(max_nodes, N), 10, key="t1_max")
    with col_ctrl3:
        t1_show_ids = st.checkbox("Node IDs", value=show_ids, key="t1_ids")

    if t1_colour != "(none)":
        all_vals = sorted(NODE_DF[t1_colour].dropna().unique().tolist(), key=str)
        t1_filter = st.multiselect(f"Filter by {t1_colour}", all_vals, default=all_vals, key="t1_filter")
    else:
        t1_filter = None

    draw_raw = st.button("🖊 Draw / Refresh Graph", key="draw_raw", use_container_width=False)

    if draw_raw or "t1_fig" not in st.session_state:
        if t1_colour != "(none)" and t1_filter:
            mask = NODE_DF[t1_colour].isin(t1_filter)
            valid = set(NODE_DF.index[mask])
        else:
            valid = set(range(N))
        subset = _get_top_nodes_by_degree(valid, t1_max)
        subset_set = set(subset)
        rows_e, cols_e = np.nonzero(np.triu(RAW_ADJ, k=1))
        edge_mask = np.isin(rows_e, list(subset_set)) & np.isin(cols_e, list(subset_set))
        edge_list = list(zip(rows_e[edge_mask].tolist(), cols_e[edge_mask].tolist()))
        n_e = len(edge_list)
        n_n = len(subset)
        deg = np.array((RAW_ADJ > 0).sum(axis=1))
        d_min = deg[subset].min() if subset else 0
        d_max = deg[subset].max() if subset else 1
        sizes = np.array([20 + 45 * (deg[nd] - d_min) / (d_max - d_min + 1e-9) for nd in subset])
        fig = _build_plotly_graph(
            node_list=subset, edge_list=edge_list, node_df=NODE_DF,
            colour_col=t1_colour, cat_cols=CAT_COLS, node_sizes=sizes,
            title=f"Raw graph · {n_n} nodes · {n_e} edges",
            show_ids=t1_show_ids, highlight_node=st.session_state.selected_node,
        )
        st.session_state["t1_fig"] = fig
        st.session_state["t1_n"] = n_n
        st.session_state["t1_e"] = n_e

    fig = st.session_state.get("t1_fig")
    n_n = st.session_state.get("t1_n", 0)
    n_e = st.session_state.get("t1_e", 0)
    st.markdown(
        f"<div class='metric-row'>"
        f"<div class='metric'><div class='metric-val'>{n_n}</div><div class='metric-lbl'>Nodes shown</div></div>"
        f"<div class='metric'><div class='metric-val'>{n_e}</div><div class='metric-lbl'>Edges shown</div></div>"
        f"</div>", unsafe_allow_html=True,
    )
    if fig:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="raw_graph_chart")
        if event and event.get("selection") and event["selection"].get("points"):
            pt = event["selection"]["points"][0]
            cd = pt.get("customdata")
            if cd is not None:
                nid = int(cd)
                if st.session_state.selected_node != nid:
                    st.session_state.selected_node = nid
                    st.info(f"✔ Node {nid} selected — switch to the **Node Inspector** tab to inspect it.")

    if st.session_state.selected_node is not None:
        nd = st.session_state.selected_node
        row = NODE_DF.iloc[nd]
        first_attr = str(row.iloc[0])[:60] if len(row) > 0 else ""
        st.markdown(
            f"<div class='card'><b style='color:var(--text-subheading)'>📌 Selected node: {nd}</b>"
            f" &nbsp; <span style='color:var(--text-secondary)'>{first_attr}</span></div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Node Inspector
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<h3 style='color:var(--text-heading)'>Node Inspector</h3>"
        "<p style='color:var(--text-secondary);font-size:13px'>"
        "Select a node (or click one in the Raw Graph tab) to see its attributes and neighbourhood."
        "</p>", unsafe_allow_html=True,
    )
    default_nid = st.session_state.selected_node if st.session_state.selected_node is not None else 0
    t2_nid = st.number_input("Node ID", min_value=0, max_value=N - 1, value=default_nid, step=1, key="t2_nid")

    if st.button("🔍 Inspect Node", key="inspect_btn", use_container_width=False):
        st.session_state.selected_node = int(t2_nid)

    nid = st.session_state.selected_node
    if nid is None:
        nid = 0
    row = NODE_DF.iloc[nid]

    st.markdown(f"<h4 style='color:var(--text-subheading)'>Node {nid} — Attributes</h4>", unsafe_allow_html=True)
    rows_html = "".join(
        f"<tr><td style='font-weight:600;color:var(--text-link);padding:6px 12px'>{col}</td>"
        f"<td style='padding:6px 12px;color:var(--text-primary)'>{_html.escape(str(row[col])[:300])}</td></tr>"
        for col in COLS
    )
    deg_all = int((RAW_ADJ[nid] > 0).sum())
    st.markdown(
        f"<div class='card'>"
        f"<p><b>Raw degree (all edges):</b> <span style='color:var(--text-accent-green)'>{deg_all}</span></p>"
        f"<div class='scroll-x'><table class='node-table'>"
        f"<tr><th>Feature</th><th>Value</th></tr>{rows_html}</table></div></div>",
        unsafe_allow_html=True,
    )

    ew = RAW_ADJ[nid].copy()
    ew[nid] = 0
    all_nbrs = np.argsort(ew)[::-1]
    all_nbrs = all_nbrs[ew[all_nbrs] > 0]
    degree = len(all_nbrs)

    if degree == 0:
        st.warning("This node has no neighbours.")
    else:
        st.markdown(f"<h4 style='color:var(--text-subheading)'>Neighbourhood — {degree} neighbours</h4>", unsafe_allow_html=True)

        bar_labels = [str(i) for i in all_nbrs[:100]]
        bar_vals = [float(ew[i]) for i in all_nbrs[:100]]
        fig_bar = go.Figure(go.Bar(
            x=bar_labels, y=bar_vals,
            marker_color=_t()["bar_marker"], marker_line_color=_t()["bar_mline"], marker_line_width=0.5,
        ))
        fig_bar.update_layout(
            title=f"Edge weights to {min(degree,100)} neighbours of node {nid}",
            paper_bgcolor=_t()["plotly_panel"], plot_bgcolor=_t()["plotly_panel"],
            font=dict(color=_t()["plotly_font"]),
            xaxis=dict(title="Neighbour node ID", tickangle=90, showgrid=False),
            yaxis=dict(title="Edge weight", gridcolor=_t()["plotly_grid"]),
            height=280, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        top_nbrs = all_nbrs[:50].tolist()
        sub_nodes = [nid] + top_nbrs
        sub_edges = [(nid, nb) for nb in top_nbrs]
        ew_list = [float(ew[nb]) for nb in top_nbrs]
        ew_arr = np.array(ew_list)
        d_min_s, d_max_s = ew_arr.min(), ew_arr.max()
        sizes_sub = np.array([35] + [16 + 28 * (ew[nb] - d_min_s) / (d_max_s - d_min_s + 1e-9) for nb in top_nbrs])

        fig_sub = _build_plotly_graph(
            node_list=sub_nodes, edge_list=sub_edges, node_df=NODE_DF,
            colour_col=colour_col, cat_cols=CAT_COLS, node_sizes=sizes_sub,
            edge_weights=ew_list,
            title=f"Neighbourhood sub-graph of node {nid} (top {len(top_nbrs)} neighbours)",
            show_ids=True, highlight_node=nid,
        )
        fig_sub.update_layout(height=640)
        st.plotly_chart(fig_sub, use_container_width=True)

        st.markdown(
            f"<h4 style='color:var(--text-subheading)'>Full Graph Context — node {nid} in the whole graph</h4>",
            unsafe_allow_html=True,
        )
        ctx_max = min(max_nodes, N)
        all_deg = np.array((RAW_ADJ > 0).sum(axis=1))
        ctx_ranked = sorted(range(N), key=lambda n: -all_deg[n])[:ctx_max]
        if nid not in set(ctx_ranked):
            ctx_ranked = [nid] + ctx_ranked[:ctx_max - 1]
        ctx_set = set(ctx_ranked)
        rows_c, cols_c = np.nonzero(np.triu(RAW_ADJ, k=1))
        emask_c = np.isin(rows_c, list(ctx_set)) & np.isin(cols_c, list(ctx_set))
        edge_list_ctx = list(zip(rows_c[emask_c].tolist(), cols_c[emask_c].tolist()))
        deg_ctx = np.array([all_deg[nd] for nd in ctx_ranked], dtype=float)
        d_min_ctx = deg_ctx.min() if len(deg_ctx) else 0
        d_max_ctx = deg_ctx.max() if len(deg_ctx) else 1
        sizes_ctx = 20 + 45 * (deg_ctx - d_min_ctx) / (d_max_ctx - d_min_ctx + 1e-9)
        for i, nd in enumerate(ctx_ranked):
            if nd == nid:
                sizes_ctx[i] = max(sizes_ctx[i], 40)
        hl_edges_ctx = [(nid, nb) for nb in top_nbrs] + [(nb, nid) for nb in top_nbrs]

        if st.session_state.get("t2_ctx_nid") != nid or "t2_ctx_fig" not in st.session_state:
            fig_ctx = _build_plotly_graph(
                node_list=ctx_ranked, edge_list=edge_list_ctx, node_df=NODE_DF,
                colour_col=colour_col, cat_cols=CAT_COLS, node_sizes=sizes_ctx,
                title=f"Full graph context — node {nid} highlighted (top {len(ctx_ranked)} nodes by degree)",
                show_ids=False, highlight_node=nid, highlight_edges=hl_edges_ctx,
            )
            fig_ctx.update_layout(height=750)
            st.session_state["t2_ctx_fig"] = fig_ctx
            st.session_state["t2_ctx_nid"] = nid
        else:
            fig_ctx = st.session_state["t2_ctx_fig"]
        st.plotly_chart(fig_ctx, use_container_width=True)

        st.markdown("<h4 style='color:var(--text-subheading)'>Neighbour Details</h4>", unsafe_allow_html=True)
        nbr_rows_html = ""
        for nb in all_nbrs[:100]:
            nbr_row = NODE_DF.iloc[nb]
            cells = (
                f"<td style='padding:5px 10px;color:var(--text-heading)'>{nb}</td>"
                f"<td style='padding:5px 10px;color:var(--text-accent-green)'>{ew[nb]:.4f}</td>"
            )
            for col in COLS:
                cells += f"<td style='padding:5px 10px;color:var(--text-primary)'>{_html.escape(str(nbr_row[col])[:120])}</td>"
            nbr_rows_html += f"<tr>{cells}</tr>"
        col_headers = "".join(f"<th>{c}</th>" for c in COLS)
        st.markdown(
            f"<div class='card scroll-x'><table class='node-table'>"
            f"<tr><th>Node</th><th>Weight</th>{col_headers}</tr>"
            f"{nbr_rows_html}</table></div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Edge Importance Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<h3 style='color:var(--text-heading)'>Edge Importance Explorer</h3>"
        "<p style='color:var(--text-secondary);font-size:13px'>"
        "Overlay GNN importance scores on the graph. Upload one or more edge importance CSV files."
        "</p>", unsafe_allow_html=True,
    )
    st.markdown("**Step 1: Upload CSV files**")
    t3_uploads = st.file_uploader(
        "Upload CSV files directly", type=["csv"], accept_multiple_files=True,
        key="t3_uploads", help="Drag and drop files here\nLimit 2GB per file • CSV",
    )
    st.markdown("**Step 2: Load files**")
    if st.button("📥 Load files", key="load_imp"):
        new_data = {}
        for uf in (t3_uploads or []):
            stem = uf.name.rsplit(".", 1)[0]
            label = stem.rsplit("_", 1)[-1]
            try:
                df_i = pd.read_csv(uf)
                new_data[label] = df_i
                st.success(f"✔ Loaded '{label}' from {uf.name} ({len(df_i)} edges)")
            except Exception as e:
                st.error(f"✘ {uf.name}: {e}")
        if new_data:
            st.session_state.imp_data.update(new_data)

    imp_data = st.session_state.imp_data
    if not imp_data:
        st.info("Load at least one edge importance CSV to proceed.")
    else:
        labels = list(imp_data.keys())
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            t3_label = st.selectbox("Target label", labels, key="t3_label")
        with c2:
            df_cur = imp_data[t3_label]
            t3_topn = st.slider("Top N edges", 1, len(df_cur), min(50, len(df_cur)), key="t3_topn")
        with c3:
            t3_colour = st.selectbox("Colour nodes by", ["(none)"] + CAT_COLS, key="t3_colour")

        t3_show_ids = st.checkbox("Show node IDs (≤150)", value=True, key="t3_show_ids")

        if st.button("🖊 Draw Importance Graph", key="draw_imp", use_container_width=False):
            df_sorted = df_cur.copy()
            score_col = next((c for c in ["importance_score", "score", "weight"] if c in df_sorted.columns), None)
            if score_col:
                df_sorted = df_sorted.sort_values(score_col, ascending=False)
            df_top = df_sorted.head(t3_topn).reset_index(drop=True)
            src_col = next((c for c in ["source_node", "src", "source"] if c in df_top.columns), df_top.columns[0])
            tgt_col = next((c for c in ["target_node", "tgt", "target"] if c in df_top.columns), df_top.columns[1])
            node_set = set()
            edge_list_imp = []
            edge_weights_imp = []
            for _, erow in df_top.iterrows():
                s = int(erow[src_col])
                t = int(erow[tgt_col])
                w = float(erow[score_col]) if score_col else 1.0
                if 0 <= s < N and 0 <= t < N:
                    edge_list_imp.append((s, t))
                    edge_weights_imp.append(w)
                    node_set.update([s, t])
            node_list_imp = list(node_set)
            deg_imp = np.array([(RAW_ADJ[nd] > 0).sum() for nd in node_list_imp], dtype=float)
            d_min_i = deg_imp.min() if len(deg_imp) else 0
            d_max_i = deg_imp.max() if len(deg_imp) else 1
            sizes_imp = 20 + 45 * (deg_imp - d_min_i) / (d_max_i - d_min_i + 1e-9)
            fig_imp = _build_plotly_graph(
                node_list=node_list_imp, edge_list=edge_list_imp, node_df=NODE_DF,
                colour_col=t3_colour, cat_cols=CAT_COLS, node_sizes=sizes_imp,
                edge_weights=edge_weights_imp,
                title=f"Edge Importance — {t3_label} · Top {t3_topn} edges",
                show_ids=t3_show_ids, highlight_node=st.session_state.selected_node,
            )
            fig_imp.update_layout(height=750)
            st.session_state["t3_fig"] = fig_imp
            if score_col:
                scores = df_top[score_col].values
                s_min, s_max = scores.min(), scores.max()
                scores_norm = (scores - s_min) / (s_max - s_min + 1e-9)
                bar_lbls = [f"{int(r[src_col])}→{int(r[tgt_col])}" for _, r in df_top.iterrows()]
                st.session_state["t3_bar"] = (bar_lbls, scores_norm, scores)
            else:
                st.session_state["t3_bar"] = None
            st.session_state["t3_edge_list"] = list(zip(df_top[src_col].tolist(), df_top[tgt_col].tolist()))
            st.session_state["t3_df_top"] = df_top

        if "t3_bar" in st.session_state and st.session_state["t3_bar"]:
            bar_lbls, scores_norm, scores_raw = st.session_state["t3_bar"]
            fig_bar3 = go.Figure(go.Bar(
                x=bar_lbls, y=scores_norm, customdata=scores_raw,
                hovertemplate="Edge: %{x}<br>Normalised: %{y:.4f}<br>Raw: %{customdata:.6f}<extra></extra>",
                marker=dict(color=scores_norm, colorscale="YlOrRd", showscale=True,
                            colorbar=dict(title="Norm. score", thickness=12)),
            ))
            fig_bar3.update_layout(
                title="Top-N edge importance scores (normalised)",
                paper_bgcolor=_t()["plotly_panel"], plot_bgcolor=_t()["plotly_panel"],
                font=dict(color=_t()["plotly_font"]),
                xaxis=dict(title="Edge (src→tgt)", tickangle=45, showgrid=False,
                           automargin=True, nticks=min(len(bar_lbls), 30)),
                yaxis=dict(title="Normalised score", range=[0, 1.05], gridcolor=_t()["plotly_grid"]),
                height=380, margin=dict(l=10, r=10, t=40, b=120),
            )
            st.plotly_chart(fig_bar3, use_container_width=True)

        if "t3_fig" in st.session_state:
            event3 = st.plotly_chart(
                st.session_state["t3_fig"], use_container_width=True,
                on_select="rerun", key="imp_graph_chart",
            )
            if event3 and event3.get("selection") and event3["selection"].get("points"):
                pt3 = event3["selection"]["points"][0]
                cd3 = pt3.get("customdata")
                if cd3 is not None:
                    nid3 = int(cd3)
                    st.session_state.selected_node = nid3
                    st.info(f"✔ Node {nid3} selected — switch to **Node Inspector** to inspect it.")

        if "t3_df_top" in st.session_state:
            st.markdown("<h4 style='color:var(--text-subheading)'>Top Edges Table</h4>", unsafe_allow_html=True)
            df_show = st.session_state["t3_df_top"].copy()
            src_col = next((c for c in ["source_node", "src", "source"] if c in df_show.columns), df_show.columns[0])
            tgt_col = next((c for c in ["target_node", "tgt", "target"] if c in df_show.columns), df_show.columns[1])
            for attr_col in COLS[:4]:
                df_show[f"src_{attr_col}"] = df_show[src_col].apply(
                    lambda x: str(NODE_DF[attr_col].iloc[int(x)])[:80] if 0 <= int(x) < N else "N/A")
                df_show[f"tgt_{attr_col}"] = df_show[tgt_col].apply(
                    lambda x: str(NODE_DF[attr_col].iloc[int(x)])[:80] if 0 <= int(x) < N else "N/A")
            st.dataframe(df_show, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LLM Explanations (GraphXAIn)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        "<h3 style='color:var(--text-heading)'>🧠 Human-Interpretable Explanations (GraphXAIn)</h3>"
        "<p style='color:var(--text-secondary);font-size:13px'>"
        "Generate natural-language explanations for the top-N important edges using a locally-running LLM via "
        "<a href='https://ollama.com' style='color:var(--text-heading)' target='_blank'>Ollama</a>."
        "</p>", unsafe_allow_html=True,
    )
    imp_data4 = st.session_state.imp_data
    if not imp_data4:
        st.warning("⚠ No importance data loaded. Go to the **Edge Importance** tab and load CSV files first.")
    else:
        c4a, c4b, c4c = st.columns([3, 2, 2])
        with c4a:
            xain_url = st.text_input("Ollama URL", value="http://localhost:11434", key="xain_url")
        with c4b:
            xain_model = st.selectbox(
                "Model", ["deepseek-r1:7b", "llama3", "mistral", "llama2", "phi3", "gemma3", "qwen2.5:7b"],
                key="xain_model",
            )
        with c4c:
            xain_label = st.selectbox("Target label", list(imp_data4.keys()), key="xain_label")

        df_xain = imp_data4[xain_label]
        score_col4 = next((c for c in ["importance_score", "score", "weight"] if c in df_xain.columns), None)
        xain_topn = st.slider("Top N edges to explain", 1, min(len(df_xain), 100), min(10, len(df_xain)), key="xain_topn")
        xain_export = st.text_input(
            "Export explanations to CSV",
            value="analysis_results/graphxain_explanations.csv", key="xain_export",
        )

        col_check, col_gen = st.columns([1, 3])
        with col_check:
            if st.button("🔌 Check Ollama", key="check_ollama"):
                try:
                    checked_url = _validate_ollama_url(xain_url)
                    r = requests.get(checked_url.rstrip("/") + "/api/tags", timeout=5)
                    models = [m["name"] for m in r.json().get("models", [])]
                    if models:
                        st.success(f"✔ Connected · {len(models)} models: {', '.join(models[:5])}")
                    else:
                        st.success("✔ Ollama is running (no models pulled yet)")
                except ValueError as ve:
                    st.error(f"✘ Invalid URL: {ve}")
                except Exception as e:
                    st.error(f"✘ Cannot connect: {e}")

        with col_gen:
            generate_clicked = st.button("⚡ Generate Explanations", key="gen_xain", type="primary", use_container_width=True)

        if generate_clicked:
            try:
                validated_ollama_url = _validate_ollama_url(xain_url)
            except ValueError as ve:
                st.error(f"✘ Invalid Ollama URL: {ve}")
                st.stop()

            df_sorted4 = df_xain.copy()
            if score_col4:
                df_sorted4 = df_sorted4.sort_values(score_col4, ascending=False)
            df_top4 = df_sorted4.head(xain_topn).reset_index(drop=True)
            src_col4 = next((c for c in ["source_node", "src", "source"] if c in df_top4.columns), df_top4.columns[0])
            tgt_col4 = next((c for c in ["target_node", "tgt", "target"] if c in df_top4.columns), df_top4.columns[1])
            text_cols4 = [c for c in NODE_DF.columns if any(k in c.lower() for k in ["text", "message", "content", "tweet", "post"])]
            label_col4 = next((c for c in NODE_DF.columns if "label" in c.lower()), None)
            ingroup_col4 = next((c for c in NODE_DF.columns if "in" in c.lower() and "group" in c.lower()), None)
            outgroup_col4 = next((c for c in NODE_DF.columns if "out" in c.lower() and "group" in c.lower()), None)

            progress_bar = st.progress(0, text="Generating explanations…")
            status_placeholder = st.empty()
            results4 = []
            error_count = 0
            ollama_url4 = validated_ollama_url.rstrip("/") + "/api/generate"

            for rank, (_, erow) in enumerate(df_top4.iterrows(), start=1):
                src4 = int(erow[src_col4])
                tgt4 = int(erow[tgt_col4])
                score4 = float(erow[score_col4]) if score_col4 else float("nan")
                src_text4 = _node_text(src4, text_cols4)
                tgt_text4 = _node_text(tgt4, text_cols4)
                src_label4 = _node_attr(src4, label_col4)
                tgt_label4 = _node_attr(tgt4, label_col4)
                src_ing4 = _node_attr(src4, ingroup_col4)
                tgt_ing4 = _node_attr(tgt4, ingroup_col4)
                src_outg4 = _node_attr(src4, outgroup_col4)
                tgt_outg4 = _node_attr(tgt4, outgroup_col4)

                prompt4 = (
                    "A Graph Neural Network identified an important connection between two messages "
                    "in a potential extremist narrative.\n\n"
                    f"Edge importance score: {score4:.6f}\n\n"
                    f"Source message (node {src4}):\n{src_text4}\n\n"
                    f"Target message (node {tgt4}):\n{tgt_text4}\n\n"
                    "Source attributes:\n"
                    f"- Label: {src_label4}\n- In-group: {src_ing4}\n- Out-group: {src_outg4}\n\n"
                    "Target attributes:\n"
                    f"- Label: {tgt_label4}\n- In-group: {tgt_ing4}\n- Out-group: {tgt_outg4}\n\n"
                    "Task:\nAnalyze the relationship between the two messages and provide a structured explanation.\n\n"
                    "You must identify:\n- the actors involved\n- the relationship between the messages\n"
                    "- the narrative type (e.g., hostility, victimization, blame, fear, identity reinforcement)\n"
                    "- why this connection is important for the model prediction\n\n"
                    "Guidelines:\n- Consider semantic similarity or reinforcement\n"
                    "- Consider in-group / out-group dynamics\n"
                    "- Consider how the messages may contribute to extremist discourse\n\n"
                    "Output format (STRICT):\nReturn ONLY a valid JSON object:\n\n"
                    '{\n  "actors": "...",\n  "relationship": "...",\n'
                    '  "narrative_type": "...",\n  "importance": "...",\n  "explanation": "..."\n}\n\n'
                    "Constraints:\n- The output MUST be valid JSON\n"
                    "- Do not add any text before or after the JSON\n"
                    "- Do not use markdown formatting\n"
                    "- Use complete and concise sentences inside each field\n"
                    "- Base your explanation ONLY on the provided messages and attributes"
                )

                raw_resp4 = ""
                try:
                    resp4 = requests.post(
                        ollama_url4,
                        json={"model": xain_model, "prompt": prompt4, "stream": False},
                        timeout=OLLAMA_TIMEOUT_SECONDS,
                    )
                    resp4.raise_for_status()
                    resp_json4 = resp4.json()
                    if "error" in resp_json4:
                        raw_resp4 = json.dumps({"error": resp_json4["error"]})
                        error_count += 1
                    else:
                        raw_resp4 = resp_json4.get("response", "")
                        if not raw_resp4:
                            raw_resp4 = json.dumps({"error": "Ollama returned an empty response."})
                            error_count += 1
                except requests.exceptions.HTTPError as he:
                    raw_resp4 = json.dumps({"error": f"HTTP {he.response.status_code}: {he.response.text[:200]}…"})
                    error_count += 1
                except requests.exceptions.Timeout:
                    raw_resp4 = json.dumps({"error": f"Timeout after {OLLAMA_TIMEOUT_SECONDS}s."})
                    error_count += 1
                except requests.exceptions.ConnectionError:
                    raw_resp4 = json.dumps({"error": f"Cannot connect to Ollama at {validated_ollama_url}."})
                    error_count += 1
                except Exception as exc:
                    raw_resp4 = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                    error_count += 1

                parsed4 = _parse_json(raw_resp4)
                results4.append({
                    "rank": rank, "source_node": src4, "target_node": tgt4,
                    "importance_score": score4, "source_text": src_text4,
                    "target_text": tgt_text4, **parsed4, "_raw_response": raw_resp4,
                })
                progress_bar.progress(rank / xain_topn, text=f"Explained {rank}/{xain_topn} edges…")

            progress_bar.empty()
            st.session_state["xain_results"] = results4
            ok_count = len(results4) - error_count
            if error_count == 0:
                status_placeholder.success(f"✔ Generated {len(results4)} explanations successfully.")
            elif ok_count > 0:
                status_placeholder.warning(f"⚠ {ok_count}/{len(results4)} explanations generated. {error_count} failed.")
            else:
                status_placeholder.error(f"✘ All {error_count} explanation calls failed.")

            if xain_export.strip():
                try:
                    export_path_safe = _safe_local_path(xain_export.strip())
                    out_rows = [{k: v for k, v in r.items() if k != "_raw_response"} for r in results4]
                    export_path_safe.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(out_rows).to_csv(export_path_safe, index=False)
                    st.success(f"✔ Explanations saved to {export_path_safe}")
                except ValueError as ve:
                    st.error(f"Invalid export path: {ve}")
                except Exception as e:
                    st.error(f"Could not save CSV: {e}")

        results_disp = st.session_state.get("xain_results", [])
        if results_disp:
            st.markdown(
                f"<h4 style='color:var(--text-subheading)'>Explanation Results — {len(results_disp)} edges</h4>",
                unsafe_allow_html=True,
            )
            present_fields = [
                f for f in ["actors", "relationship", "narrative_type", "importance", "explanation", "error"]
                if any(f in r for r in results_disp)
            ]

            def _td4(content, style=""):
                safe = _html.escape(str(content))
                return f'<td style="padding:7px 10px;border:1px solid var(--td-border);vertical-align:top;color:var(--text-primary){style}">{safe}</td>'

            def _trunc4(text, n=120):
                return (text[:n] + "…") if len(text) > n else text

            rows_html4 = ""
            for r in results_disp:
                cells4 = _td4(f"#{r['rank']}", ";text-align:center;font-weight:bold;color:var(--text-subheading)")
                cells4 += _td4(f"{r['source_node']} → {r['target_node']}", ";text-align:center;color:var(--text-heading)")
                cells4 += _td4(f"{r['importance_score']:.6f}", ";text-align:right;color:var(--text-accent-green)")
                cells4 += _td4(_trunc4(r["source_text"]), ";color:var(--text-secondary);font-size:11px")
                cells4 += _td4(_trunc4(r["target_text"]), ";color:var(--text-secondary);font-size:11px")
                for f in present_fields:
                    cells4 += _td4(r.get(f, "—"))
                bg = "var(--xain-row-even)" if r["rank"] % 2 == 0 else "var(--xain-row-odd)"
                rows_html4 += f'<tr style="background:{bg}">{cells4}</tr>\n'

            th_s = 'style="padding:9px 12px;background:var(--xain-th-bg);border:1px solid var(--xain-th-border);color:var(--text-subheading)"'
            fixed_ths = "".join(f"<th {th_s}>{h}</th>" for h in ["#", "Edge", "Score", "Source text", "Target text"])
            field_ths = "".join(f'<th {th_s}>{f.replace("_", " ").title()}</th>' for f in present_fields)
            st.markdown(
                "<div class='scroll-x' style='margin-top:10px'>\n<table class='xain-table'>\n"
                f"  <thead><tr>{fixed_ths}{field_ths}</tr></thead>\n"
                f"  <tbody>{rows_html4}</tbody>\n</table></div>",
                unsafe_allow_html=True,
            )

            csv_bytes = pd.DataFrame([
                {k: v for k, v in r.items() if k != "_raw_response"} for r in results_disp
            ]).to_csv(index=False).encode()
            st.download_button(
                "⬇ Download explanations CSV", data=csv_bytes,
                file_name="graphxain_explanations.csv", mime="text/csv", key="dl_xain",
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border-color:var(--border-footer);margin-top:40px'>"
    "<p style='color:var(--text-footer);font-size:11px;text-align:center'>"
    "GR-EN-A-DE Explorer · Built with Streamlit + Plotly · GraphXAIn via Ollama"
    "</p>",
    unsafe_allow_html=True,
)
