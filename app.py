"""Streamlit entry point: upload -> validate -> clean -> KPIs -> EDA charts."""
from pathlib import Path

import requests
import streamlit as st

from src.analysis.charts import (
    category_bar_chart,
    correlation_heatmap,
    histogram_chart,
    missing_values_chart,
    time_series_chart,
)
from src.analysis.eda import aggregate_over_time, run_eda
from src.analysis.kpis import compute_kpis
from src.analysis.metric_selectors import pick_primary_date_column, pick_primary_metrics
from src.cleaning.cleaner import (
    CleaningOptions,
    CleaningResult,
    clean_dataframe,
    log_to_dataframe,
)
from src.config import settings
from src.ingestion.loader import (
    EXCEL_EXTENSIONS,
    FileLoadError,
    LoadResult,
    get_excel_sheet_names,
    load_file,
)
from src.ingestion.validator import validate_dataframe
from src.profiling.type_detector import profile_columns, profiles_to_dataframe

st.set_page_config(page_title="AI-Powered Automated Data Analyst", page_icon="📊", layout="wide")

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


@st.cache_data(show_spinner=False)
def cached_load(data: bytes, filename: str, sheet_name: str | None) -> LoadResult:
    return load_file(data, filename, sheet_name)


@st.cache_data(show_spinner=False)
def cached_clean(df, options: CleaningOptions) -> CleaningResult:
    return clean_dataframe(df, options)


@st.cache_data(show_spinner=False)
def cached_analysis(df, profiles):
    return run_eda(df, profiles), compute_kpis(df, profiles)


def check_ollama() -> tuple[bool, str]:
    try:
        resp = requests.get(f"{settings.ollama_host}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if not models:
            return True, "Ollama is running, but no models are downloaded yet."
        return True, "Ollama is running. Models: " + ", ".join(models)
    except requests.RequestException as exc:
        return False, f"Cannot reach Ollama ({exc.__class__.__name__})"


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("System status")
    ok, message = check_ollama()
    (st.success if ok else st.error)(message)
    st.caption(f"AI model: {settings.ollama_model}")
    st.caption(f"Max upload: {settings.max_upload_mb} MB")

    st.header("Cleaning options")
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

# ---------------- Main page ----------------
st.title("📊 AI-Powered Automated Data Analyst")
st.caption("Upload a CSV or Excel file to begin.")

st.subheader("1. Upload your data")
uploaded = st.file_uploader(
    "Choose a file", type=["csv", "tsv", "txt", "xlsx", "xlsm"], help="CSV, TSV, TXT or Excel"
)
if uploaded is None:
    st.info("Waiting for a file. Sample files are in the data/sample folder of the project.")
    st.stop()

data = uploaded.getvalue()

sheet_choice = None
if Path(uploaded.name).suffix.lower() in EXCEL_EXTENSIONS:
    try:
        sheets = get_excel_sheet_names(data)
    except FileLoadError as exc:
        st.error(str(exc))
        st.stop()
    sheet_choice = (
        st.selectbox("This workbook has several sheets. Pick one:", sheets)
        if len(sheets) > 1
        else sheets[0]
    )

try:
    result = cached_load(data, uploaded.name, sheet_choice)
except FileLoadError as exc:
    st.error(str(exc))
    st.stop()

df = result.df
report = validate_dataframe(df)

# ---------------- File summary ----------------
st.subheader("2. File summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{report.info['rows']:,}")
c2.metric("Columns", report.info["columns"])
c3.metric("Size", f"{result.size_bytes / 1024:,.0f} KB")
c4.metric("Missing cells", f"{report.info['missing_pct']}%")

details = [f"**Type:** {result.file_type}"]
if result.encoding:
    details.append(f"**Encoding:** {result.encoding}")
if result.delimiter:
    shown = {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}.get(
        result.delimiter, result.delimiter
    )
    details.append(f"**Delimiter:** {shown}")
if result.sheet_name:
    details.append(f"**Sheet:** {result.sheet_name}")
st.markdown("  |  ".join(details))
for note in result.notes:
    st.warning(note)

# ---------------- Validation ----------------
st.subheader("3. Validation")
if report.errors:
    for err in report.errors:
        st.error(err)
    st.stop()
st.success("The file passed validation and can be analyzed.")
if report.warnings:
    with st.expander(f"{len(report.warnings)} warning(s)", expanded=False):
        for warning in report.warnings:
            st.warning(warning)

# ---------------- Detection + cleaning ----------------
raw_profiles = profile_columns(df)
cleaning = cached_clean(df, options)
cleaned = cleaning.df

st.subheader("4. Types and cleaning")
tab_raw, tab_types, tab_log, tab_clean = st.tabs(
    ["Raw preview", "Detected column types", "Cleaning report", "Cleaned data"]
)

with tab_raw:
    st.dataframe(df.head(100))

with tab_types:
    st.caption("Types detected on the raw file. 'stored as text' means the values were converted.")
    st.dataframe(profiles_to_dataframe(raw_profiles))

with tab_log:
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", f"{cleaning.after_shape[0]:,}", f"{cleaning.after_shape[0] - cleaning.before_shape[0]:,}")
    m2.metric("Columns", cleaning.after_shape[1], cleaning.after_shape[1] - cleaning.before_shape[1])
    m3.metric(
        "Missing cells",
        f"{int(cleaned.isna().sum().sum()):,}",
        f"{int(cleaned.isna().sum().sum()) - int(df.isna().sum().sum()):,}",
        delta_color="inverse",
    )
    if cleaning.log:
        st.dataframe(log_to_dataframe(cleaning.log))
    else:
        st.info("Nothing needed cleaning. The data was already tidy.")

with tab_clean:
    st.caption("Final column types after cleaning:")
    st.dataframe(profiles_to_dataframe(cleaning.profiles))
    st.caption("Cleaned data (first 100 rows):")
    st.dataframe(cleaned.head(100))
    st.download_button(
        "Download cleaned data (CSV)",
        cleaned.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

# ---------------- KPIs ----------------
eda, kpis = cached_analysis(cleaned, cleaning.profiles)

st.subheader("5. Key metrics (KPIs)")
for start in range(0, len(kpis), 4):
    cols = st.columns(4)
    for widget, kpi in zip(cols, kpis[start:start + 4]):
        widget.metric(kpi.name, kpi.display, help=kpi.description)

# ---------------- EDA ----------------
st.subheader("6. Automated exploration")
t_dist, t_cat, t_corr, t_time, t_quality = st.tabs(
    ["Distributions", "Categories", "Correlations", "Over time", "Data quality"]
)

with t_dist:
    if not eda.numeric_columns:
        st.info("No numeric columns were found.")
    else:
        num_col = st.selectbox("Numeric column", eda.numeric_columns, key="dist_col")
        st.plotly_chart(histogram_chart(cleaned, num_col), key="dist_chart")
        with st.expander("Summary statistics for all numeric columns"):
            st.dataframe(eda.numeric_summary)

with t_cat:
    if not eda.categorical_summaries:
        st.info("No categorical columns were found.")
    else:
        cat_col = st.selectbox("Categorical column", list(eda.categorical_summaries), key="cat_col")
        st.plotly_chart(category_bar_chart(cleaned, cat_col), key="cat_chart")
        st.dataframe(eda.categorical_summaries[cat_col])

with t_corr:
    if eda.correlation_matrix is None:
        st.info("At least two numeric columns are needed for correlations.")
    else:
        st.plotly_chart(correlation_heatmap(eda.correlation_matrix), key="corr_chart")
        if eda.strong_correlations:
            for pair in eda.strong_correlations[:10]:
                st.write(
                    f"- **{pair.column_a}** and **{pair.column_b}**: "
                    f"{pair.strength} {pair.direction} relationship (r = {pair.correlation})"
                )
            st.caption("Correlation means two columns move together. It does not prove that one causes the other.")
        else:
            st.info("No strong correlations (|r| of 0.6 or more) were found.")

with t_time:
    if not eda.datetime_columns or not eda.numeric_columns:
        st.info("A date column and at least one numeric column are needed for a time chart.")
    else:
        default_date = pick_primary_date_column(cleaning.profiles) or eda.datetime_columns[0]
        date_col = st.selectbox(
            "Date column", eda.datetime_columns, index=eda.datetime_columns.index(default_date), key="ts_date"
        )
        default_metric = pick_primary_metrics(cleaning.profiles, k=1)[0]
        metric = st.selectbox(
            "Metric", eda.numeric_columns, index=eda.numeric_columns.index(default_metric), key="ts_metric"
        )
        ts, period_label = aggregate_over_time(cleaned, date_col, metric)
        if ts.empty:
            st.info("There is no data for this combination.")
        else:
            st.plotly_chart(time_series_chart(ts, metric, period_label), key="ts_chart")
        for info in eda.datetime_info:
            if info["column"] == date_col:
                st.caption(
                    f"{date_col}: {info['start']} to {info['end']} ({info['span_days']:,} days), "
                    f"{info['frequency']} data."
                )

with t_quality:
    fig = missing_values_chart(raw_profiles)
    if fig is None:
        st.success("No missing values were found in the original file.")
    else:
        st.plotly_chart(fig, key="missing_chart")
    st.caption("Based on the raw file, before cleaning.")

# Keep results available for the next phases
st.session_state["load_result"] = result
st.session_state["cleaning_result"] = cleaning
st.session_state["eda_result"] = eda
st.session_state["kpis"] = kpis
