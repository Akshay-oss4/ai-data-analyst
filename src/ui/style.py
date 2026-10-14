"""Look and feel: CSS for the dashboard and the header bar. Purely cosmetic; no logic lives here.

To change the colors, edit --brand and --brand-2 below (and the gradient colors in .topbar and .hero).
"""
from __future__ import annotations

from html import escape

import streamlit as st

CSS = """
:root {
  --brand: #4f46e5;
  --brand-2: #0d9488;
  --muted: #64748b;
  --line: rgba(127, 127, 127, 0.28);
  --card: rgba(127, 127, 127, 0.07);
}

/* ---------- page spacing and headings ---------- */
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }
footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.01em; }
h2 { border-bottom: 2px solid var(--line); padding-bottom: 0.35rem; margin-top: 0.4rem; }

/* ---------- header bar ---------- */
.topbar {
  display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
  padding: 14px 20px; margin-bottom: 1rem; border-radius: 14px; color: #fff;
  background: linear-gradient(120deg, #4f46e5, #0d9488);
}
.topbar-title { font-size: 1.15rem; font-weight: 700; }
.topbar-sub { font-size: 0.85rem; opacity: 0.9; }
.chip {
  display: inline-block; background: rgba(255, 255, 255, 0.22); border-radius: 999px;
  padding: 2px 11px; font-size: 0.78rem; margin-left: 0.4rem;
}

/* ---------- welcome screen ---------- */
.hero {
  padding: 28px; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;
  background: linear-gradient(120deg, #4f46e5, #0d9488);
}
.hero h1 { color: #fff; margin: 0 0 0.4rem 0; padding: 0; }
.hero p { margin: 0; font-size: 1.05rem; opacity: 0.95; }
.feature { border: 1px solid var(--line); background: var(--card); border-radius: 14px; padding: 18px; height: 100%; }
.feature-icon { font-size: 1.8rem; }
.feature h4 { margin: 0.3rem 0; }
.feature p { color: var(--muted); margin: 0; }

/* ---------- KPI tiles ---------- */
[data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--line); border-top: 3px solid var(--brand);
  border-radius: 12px; padding: 14px 16px;
  transition: box-shadow 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12); transform: translateY(-1px); }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.82rem; }
[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 650; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h1 { font-size: 1.35rem; color: var(--brand); }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 0.45rem 0.7rem; border-radius: 10px; width: 100%; transition: background 0.15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(79, 70, 229, 0.10); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(79, 70, 229, 0.14); box-shadow: inset 3px 0 0 var(--brand); font-weight: 600;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; }
.stTabs [aria-selected="true"] { color: var(--brand); font-weight: 600; }

/* ---------- buttons, upload zone, tables, boxes ---------- */
.stButton > button, [data-testid="stDownloadButton"] > button {
  border-radius: 10px; border: 1px solid var(--brand); font-weight: 600; transition: all 0.15s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
  background: var(--brand); border-color: var(--brand); color: #fff;
}
.stButton > button:hover p, [data-testid="stDownloadButton"] > button:hover p { color: #fff; }
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed rgba(79, 70, 229, 0.45); border-radius: 12px; background: rgba(79, 70, 229, 0.04);
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; }
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }

/* ---------- finding cards ---------- */
.finding {
  border: 1px solid var(--line); border-left: 5px solid var(--brand); background: var(--card);
  border-radius: 10px; padding: 9px 12px; margin-bottom: 0.55rem;
}
.finding-tag { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; color: var(--muted); margin-right: 0.35rem; }
.finding-detail { font-size: 0.9rem; color: var(--muted); margin-top: 2px; }
"""


def inject_css() -> None:
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def build_top_bar_html(filename: str, shape: tuple[int, int], filters_applied: list[str], demo: bool) -> str:
    """The header bar. Everything that comes from the user's data is HTML-escaped."""
    rows, cols = shape
    chips = []
    if filters_applied:
        label = f"🔎 {len(filters_applied)} filter" + ("s" if len(filters_applied) > 1 else "") + " active"
        title = escape("; ".join(filters_applied), quote=True)
        chips.append(f'<span class="chip" title="{title}">{escape(label)}</span>')
    if demo:
        chips.append('<span class="chip">Demo mode</span>')
    return (
        '<div class="topbar"><div>'
        '<div class="topbar-title">📊 AI-Powered Automated Data Analyst</div>'
        f'<div class="topbar-sub">{escape(filename)} · {rows:,} rows × {cols} columns</div>'
        f'</div><div>{"".join(chips)}</div></div>'
    )


def top_bar(filename: str, shape: tuple[int, int], filters_applied: list[str], demo: bool) -> None:
    st.markdown(build_top_bar_html(filename, shape, filters_applied, demo), unsafe_allow_html=True)