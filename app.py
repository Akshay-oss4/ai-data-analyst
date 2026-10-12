"""Streamlit entry point.

Sidebar: navigation, upload, cleaning options, filters, system status.
Main area: the selected page (Overview, Data, Analysis, Insights, Report, History).
"""
import hashlib
import traceback
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai import llm
from src.ai.ollama_client import list_models
from src.analysis.metric_selectors import pick_primary_date_column
from src.analysis.pipeline import AnalysisBundle, run_analysis
from src.cleaning.cleaner import CleaningOptions, CleaningResult, clean_dataframe
from src.config import settings
from src.ingestion.loader import (
    EXCEL_EXTENSIONS,
    FileLoadError,
    LoadResult,
    get_excel_sheet_names,
    load_file,
)
from src.ingestion.validator import ValidationReport, validate_dataframe
from src.logging_setup import get_logger
from src.profiling.type_detector import CATEGORICAL, ColumnProfile, profile_columns
from src.ui.filters import FilterSpec, apply_filters, refresh_profiles
from src.ui.history_page import render_history_page, run_auto_save
from src.ui.pages import (
    AppContext,
    render_analysis,
    render_data,
    render_insights_page,
    render_overview,
    render_welcome,
)
from src.ui.report_page import render_report_page

st.set_page_config(
    page_title="AI-Powered Automated Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Card look for KPI tiles (works in light and dark themes)
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = {
    "🏠 Overview": render_overview,
    "🗂️ Data": render_data,
    "📈 Analysis": render_analysis,
    "💡 Insights": render_insights_page,
    "📄 Report": render_report_page,
    "🕘 History": render_history_page,
}
NUMERIC_FILL_LABELS = {
    "median": "Fill with median",
    "mean": "Fill with mean",
    "zero": "Fill with 0",
    "none": "Leave missing",
}
CATEGORICAL_FILL_LABELS = {
    "unknown": "Fill with 'Unknown'",
    "mode": "Fill with most common value",
    "none": "Leave missing",
}

logger = get_logger()


def show_failure(stage: str, filename: str) -> None:
    """Friendly message for an unexpected error. Call it from inside an except block."""
    logger.exception("%s failed for %s", stage, filename)
    st.error(
        f"Something went wrong while {stage}. The file may have an unusual structure. "
        "The details were saved to logs/app.log."
    )
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
    st.stop()


# Arguments that start with an underscore are not hashed by Streamlit. The cache_key stands in for them,
# so a big table is not re-hashed on every click.
@st.cache_data(show_spinner=False)
def cached_load(data: bytes, filename: str, sheet_name: str | None) -> LoadResult:
    return load_file(data, filename, sheet_name)


@st.cache_data(show_spinner=False)
def cached_validate(_df, cache_key: str) -> ValidationReport:
    return validate_dataframe(_df)


@st.cache_data(show_spinner=False)
def cached_profiles(_df, cache_key: str) -> list[ColumnProfile]:
    return profile_columns(_df)


@st.cache_data(show_spinner="Cleaning your data...")
def cached_clean(_df, cache_key: str, options: CleaningOptions) -> CleaningResult:
    return clean_dataframe(_df, options)


@st.cache_data(show_spinner="Analyzing your data...")
def cached_bundle(_cleaning: CleaningResult, cache_key: str) -> AnalysisBundle:
    return run_analysis(_cleaning)


# ================= Sidebar, part 1: navigation, upload, cleaning options =================
with st.sidebar:
    st.title("📊 Data Analyst")
    page_name = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.divider()

    st.subheader("Your data")
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "tsv", "txt", "xlsx", "xlsm"],
        label_visibility="collapsed",
    )

    sheet_choice = None
    if uploaded is not None and Path(uploaded.name).suffix.lower() in EXCEL_EXTENSIONS:
        try:
            sheets = get_excel_sheet_names(uploaded.getvalue())
        except FileLoadError as exc:
            st.error(str(exc))
            st.stop()
        sheet_choice = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]

    with st.expander("Cleaning options"):
        opt_dups = st.checkbox("Remove duplicate rows", value=True)
        opt_numeric = st.selectbox(
            "Missing numbers", list(NUMERIC_FILL_LABELS), format_func=NUMERIC_FILL_LABELS.get
        )
        opt_categorical = st.selectbox(
            "Missing categories", list(CATEGORICAL_FILL_LABELS), format_func=CATEGORICAL_FILL_LABELS.get
        )
        opt_sparse = st.slider("Drop columns with more than this % missing", 10, 100, 60, step=5)

options = CleaningOptions(
    drop_duplicates=opt_dups,
    numeric_fill=opt_numeric,
    categorical_fill=opt_categorical,
    drop_column_missing_pct=float(opt_sparse),
)

if uploaded is None:
    with st.sidebar:
        st.caption("Upload a file to unlock the dashboard.")
    render_welcome()
    st.stop()

# ================= Load, validate, clean =================
data = uploaded.getvalue()
base_key = f"{hashlib.sha256(data).hexdigest()[:16]}|{sheet_choice}"

try:
    load_result = cached_load(data, uploaded.name, sheet_choice)
except FileLoadError as exc:
    st.error(str(exc))
    st.stop()
except Exception:  # noqa: BLE001 - show a friendly message instead of a traceback
    show_failure("reading the file", uploaded.name)

try:
    report = cached_validate(load_result.df, base_key)
    if not report.errors:
        raw_profiles = cached_profiles(load_result.df, base_key)
        full_cleaning = cached_clean(load_result.df, base_key, options)
except Exception:  # noqa: BLE001
    show_failure("checking and cleaning the data", uploaded.name)

if report.errors:
    for err in report.errors:
        st.error(err)
    st.stop()
if full_cleaning.after_shape[0] == 0 or full_cleaning.after_shape[1] == 0:
    st.error("Nothing is left to analyze after cleaning: every row or column was empty.")
    st.stop()

file_key = f"{uploaded.name}_{sheet_choice}"

# ================= Sidebar, part 2: filters and status =================
spec = FilterSpec()
with st.sidebar:
    with st.expander("Filters", expanded=False):
        date_col = pick_primary_date_column(full_cleaning.profiles)
        if date_col:
            dates = full_cleaning.df[date_col].dropna()
            low, high = dates.min().date(), dates.max().date()
            picked = st.date_input(
                f"{date_col} range", value=(low, high), min_value=low, max_value=high,
                key=f"f_date_{file_key}",
            )
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                spec.date_column = date_col
                spec.start, spec.end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])

        category_profiles = [
            p for p in full_cleaning.profiles
            if p.inferred_type == CATEGORICAL and 2 <= p.unique_count <= 30
        ]
        for p in sorted(category_profiles, key=lambda p: p.unique_count)[:3]:
            choices = sorted(full_cleaning.df[p.name].astype(str).unique())
            selected = st.multiselect(p.name, choices, key=f"f_{file_key}_{p.name}")
            if selected:
                spec.categories[p.name] = selected
        if not date_col and not category_profiles:
            st.caption("No date or category columns to filter on.")
        else:
            st.caption("Leave a filter empty to include everything.")

    with st.expander("System status"):
        installed = list_models()
        if installed is None:
            st.error("Cannot reach Ollama. Start the Ollama app.")
        elif not installed:
            st.warning("Ollama is running, but no model is downloaded.")
        else:
            st.success("Ollama is running")
        st.caption(f"AI provider: {llm.active_provider()}")
        st.caption(f"Max upload: {settings.max_upload_mb} MB")

# ================= Apply filters and analyze =================
filtered_df, filters_applied = apply_filters(full_cleaning.df, spec)
if filtered_df.empty:
    st.error("No rows match the current filters. Widen the date range or clear a category filter.")
    st.stop()
if filters_applied and len(filtered_df) < 30:
    st.warning(f"Only {len(filtered_df)} rows match the filters. Statistics on so few rows are unreliable.")

if filters_applied:
    cleaning = replace(
        full_cleaning,
        df=filtered_df,
        profiles=refresh_profiles(full_cleaning.profiles, filtered_df),
        after_shape=filtered_df.shape,
    )
else:
    cleaning = full_cleaning

try:
    bundle = cached_bundle(cleaning, f"{base_key}|{options}|{'; '.join(filters_applied)}")
except Exception:  # noqa: BLE001
    show_failure("analyzing the data", uploaded.name)

scope_label = uploaded.name + (f" (filtered: {'; '.join(filters_applied)})" if filters_applied else "")
ctx = AppContext(
    filename=uploaded.name,
    sheet=sheet_choice,
    options=options,
    load=load_result,
    report=report,
    raw_profiles=raw_profiles,
    full_cleaning=full_cleaning,
    cleaning=cleaning,
    bundle=bundle,
    filters_applied=filters_applied,
    scope_label=scope_label,
    insights_key=f"{scope_label}|{base_key}|{options}",
)

st.session_state["ctx"] = ctx

try:
    saved = run_auto_save(ctx)
    if saved is not None and saved[1]:
        st.toast(f"Saved to history as run #{saved[0]}", icon="🕘")
except Exception:  # noqa: BLE001 - history must never break the dashboard
    logger.exception("auto-save to history failed")

try:
    PAGES[page_name](ctx)
except Exception:  # noqa: BLE001
    logger.exception("page %s failed for %s", page_name, uploaded.name)
    st.error(
        f"The {page_name.split(' ', 1)[1]} page hit an unexpected problem. The other pages still work. "
        "The details were saved to logs/app.log."
    )
    with st.expander("Technical details"):
        st.code(traceback.format_exc())