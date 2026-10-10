"""The Report page: build the HTML report and Excel workbook, then offer them for download."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.ai.insights import generate_insights
from src.config import settings
from src.reporting.excel_report import build_excel_report
from src.reporting.html_report import build_html_report
from src.ui.pages import AppContext

CHART_MODES = {
    "inline": "Interactive charts (works offline, about 5 MB)",
    "none": "No charts (small file)",
}
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def render_report_page(ctx: AppContext) -> None:
    st.header("Report")
    st.markdown(
        "Build a shareable report of everything on the dashboard. You get a **self-contained HTML report** "
        "and an **Excel workbook** with the same results."
    )
    if ctx.filters_applied:
        st.info("Filters active: " + "; ".join(ctx.filters_applied) + ". The report covers this filtered view.")

    stored = st.session_state.get("insights")
    ai_ready = bool(stored and stored[0] == ctx.insights_key)
    if ai_ready:
        st.success("The AI insights you generated will be included in the report.")
    else:
        st.info(
            "No AI insights yet, so the report will use the automatic (rule-based) insights. "
            "For AI-written text, generate it on the Insights page first."
        )

    mode = st.radio("Charts in the HTML report", list(CHART_MODES), format_func=CHART_MODES.get, key="report_charts")
    state_key = f"{ctx.insights_key}|{mode}|{ai_ready}"

    if st.button("Build report", key="build_report"):
        with st.spinner("Building the report..."):
            insights = (
                stored[1]
                if ai_ready
                else generate_insights(ctx.bundle, ctx.cleaning, ctx.scope_label, use_llm=False)
            )
            common = dict(
                title=f"Data analysis report: {ctx.filename}",
                scope_label=ctx.scope_label,
                cleaning=ctx.cleaning,
                full_cleaning=ctx.full_cleaning,
                bundle=ctx.bundle,
                insights=insights,
                filters_applied=ctx.filters_applied,
            )
            html_bytes = build_html_report(
                **common, raw_profiles=ctx.raw_profiles, chart_mode=mode
            ).encode("utf-8")
            xlsx_bytes = build_excel_report(**common)

            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(ctx.filename).stem) or "report"
            stem = f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
            saved_to = ""
            try:  # keep a copy in the project's reports folder
                settings.reports_dir.mkdir(parents=True, exist_ok=True)
                (settings.reports_dir / f"{stem}.html").write_bytes(html_bytes)
                (settings.reports_dir / f"{stem}.xlsx").write_bytes(xlsx_bytes)
                saved_to = str(settings.reports_dir)
            except OSError:
                st.warning("Could not save a copy in the reports folder, but the downloads below still work.")

        st.session_state["report"] = {
            "key": state_key, "html": html_bytes, "xlsx": xlsx_bytes, "stem": stem, "saved_to": saved_to,
        }

    built = st.session_state.get("report")
    if built and built["key"] == state_key:
        st.success("Report ready.")
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download HTML report", built["html"], file_name=f"{built['stem']}.html",
            mime="text/html", key="dl_html",
        )
        c2.download_button(
            "Download Excel workbook", built["xlsx"], file_name=f"{built['stem']}.xlsx",
            mime=XLSX_MIME, key="dl_xlsx",
        )
        st.caption(
            "To get a PDF: open the HTML file in Chrome or Edge, press Ctrl+P, and choose 'Save as PDF'."
        )
        if built["saved_to"]:
            st.caption(f"Copies were also saved in: {built['saved_to']}")