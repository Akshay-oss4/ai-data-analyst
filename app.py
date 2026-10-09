"""Streamlit entry point: upload -> validate -> detect types -> clean -> download."""
from pathlib import Path

import requests
import streamlit as st

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
st.caption("Upload a CSV or Excel file, or start with a sample dataset.")

st.subheader("1. Upload your data")
sample_dir = Path(__file__).resolve().parent / "data" / "sample"
sample_files = sorted(
    path for path in sample_dir.iterdir()
    if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xlsm"}
)
data_source = st.radio("Data source", ["Upload a file", "Sample dataset"], horizontal=True)

if data_source == "Upload a file":
    uploaded = st.file_uploader(
        "Choose a file", type=["csv", "tsv", "txt", "xlsx", "xlsm"], help="CSV, TSV, TXT or Excel"
    )
    if uploaded is None:
        st.info("Choose a file to begin, or select a sample dataset.")
        st.stop()
    filename = uploaded.name
    data = uploaded.getvalue()
else:
    if not sample_files:
        st.warning("No sample datasets were found in data/sample.")
        st.stop()
    default_sample = next(
        (index for index, path in enumerate(sample_files) if path.name == "messy_data.csv"), 0
    )
    sample_path = st.selectbox(
        "Choose a sample dataset",
        sample_files,
        index=default_sample,
        format_func=lambda path: path.stem.replace("_", " ").title(),
    )
    filename = sample_path.name
    data = sample_path.read_bytes()

sheet_choice = None
if Path(filename).suffix.lower() in EXCEL_EXTENSIONS:
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
    result = cached_load(data, filename, sheet_choice)
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

# Keep results available for the next phases
st.session_state["load_result"] = result
st.session_state["cleaning_result"] = cleaning