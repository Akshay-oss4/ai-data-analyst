"""Streamlit entry point: upload -> load -> validate -> preview."""
import pandas as pd
import requests
import streamlit as st

from src.config import settings
from src.ingestion.loader import (
    EXCEL_EXTENSIONS,
    FileLoadError,
    LoadResult,
    get_excel_sheet_names,
    load_file,
)
from src.ingestion.validator import validate_dataframe
from pathlib import Path

st.set_page_config(page_title="AI-Powered Automated Data Analyst", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def cached_load(data: bytes, filename: str, sheet_name: str | None) -> LoadResult:
    return load_file(data, filename, sheet_name)


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

# Let the user pick a sheet for Excel workbooks
sheet_choice = None
if Path(uploaded.name).suffix.lower() in EXCEL_EXTENSIONS:
    try:
        sheets = get_excel_sheet_names(data)
    except FileLoadError as exc:
        st.error(str(exc))
        st.stop()
    if len(sheets) > 1:
        sheet_choice = st.selectbox("This workbook has several sheets. Pick one:", sheets)
    else:
        sheet_choice = sheets[0]

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
    with st.expander(f"{len(report.warnings)} warning(s)", expanded=True):
        for warning in report.warnings:
            st.warning(warning)

# ---------------- Preview ----------------
st.subheader("4. Data preview")
st.dataframe(df.head(100))

st.subheader("5. Column overview")
overview = pd.DataFrame(
    {
        "column": df.columns.astype(str),
        "pandas dtype": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "unique values": df.nunique().values,
    }
)
st.dataframe(overview)

# Keep the loaded data available for the next phases
st.session_state["load_result"] = result