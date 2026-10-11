"""The History page: save analyses, browse them, compare two runs, chart a KPI over time."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai.insights import generate_insights
from src.database.history import build_run_record, get_store
from src.ui.pages import AppContext
from src.ui.presenters import SEVERITY_ICONS


def _current_record(ctx: AppContext):
    stored = st.session_state.get("insights")
    if stored and stored[0] == ctx.insights_key:
        insights = stored[1]
    else:
        insights = generate_insights(ctx.bundle, ctx.cleaning, ctx.scope_label, use_llm=False)
    return build_run_record(
        ctx.filename, ctx.sheet, ctx.scope_label, ctx.filters_applied, ctx.cleaning, ctx.bundle, insights
    )


def run_auto_save(ctx: AppContext) -> tuple[int, bool] | None:
    """Save the unfiltered analysis once per session if automatic saving is switched on."""
    cfg = st.session_state.get("history_cfg")
    if not cfg or not cfg.get("auto") or ctx.filters_applied:
        return None
    attempted = st.session_state.setdefault("history_auto_attempted", set())
    if ctx.insights_key in attempted:
        return None
    attempted.add(ctx.insights_key)
    return get_store().save_run(_current_record(ctx))


def _label(run: dict) -> str:
    note = " (filtered)" if run["filters"] else ""
    return f"#{run['id']} · {run['created_at'].replace('T', ' ')} · {run['filename']}{note}"


def render_history_page(ctx: AppContext) -> None:
    store = get_store()
    st.header("History")
    st.markdown(
        "Save an analysis to compare it with later ones, for example this month's export against last month's."
    )

    c1, c2 = st.columns([1, 2])
    if c1.button("Save this analysis to history", key="hist_save"):
        run_id, created = store.save_run(_current_record(ctx))
        if created:
            st.success(f"Saved as run #{run_id}.")
        else:
            st.info(f"An identical analysis is already saved as run #{run_id}.")
    auto = c2.checkbox(
        "Save automatically when a new (unfiltered) file is analyzed",
        key="hist_auto",
        help="Saves each distinct analysis once. Takes effect after you have opened this page.",
    )
    st.session_state["history_cfg"] = {"auto": auto}  # widgets on other pages are forgotten, so keep a copy

    runs = store.list_runs()
    if not runs:
        st.info("No analyses are saved yet. Click the button above to save the current one.")
        return

    st.subheader("Saved analyses")
    st.dataframe(
        pd.DataFrame(runs).rename(
            columns={
                "id": "run", "created_at": "saved at", "n_rows": "rows", "n_columns": "columns",
                "n_findings": "findings", "n_high": "high priority", "summary_source": "summary by",
            }
        )[["run", "saved at", "filename", "filters", "rows", "columns", "findings", "high priority", "summary by"]]
    )

    by_label = {_label(r): r["id"] for r in runs}  # newest first
    labels = list(by_label)
    tab_open, tab_compare, tab_trend, tab_manage = st.tabs(
        ["Open a run", "Compare two runs", "KPI over time", "Manage"]
    )

    with tab_open:
        run = store.get_run(by_label[st.selectbox("Run", labels, key="hist_open")])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{run['n_rows']:,}")
        m2.metric("Columns", run["n_columns"])
        m3.metric("Findings", run["n_findings"], f"{run['n_high']} high", delta_color="off")
        m4.metric("Summary by", run["summary_source"] or "n/a")
        if run["filters"]:
            st.info("Filtered view: " + run["filters"])
        st.info(run["summary"])
        st.markdown("**KPIs**")
        st.dataframe(pd.DataFrame(run["kpis"])[["name", "display", "kind"]])
        st.markdown("**Findings**")
        for f in run["findings"]:
            st.markdown(f"{SEVERITY_ICONS.get(f['severity'], '')} **{f['title']}** ({f['severity']})  \n{f['detail']}")
        if run["trends"]:
            st.markdown("**Trends**")
            st.dataframe(pd.DataFrame(run["trends"]))

    with tab_compare:
        if len(labels) < 2:
            st.info("Save at least two analyses to compare them.")
        else:
            a_label = st.selectbox("Older run (A)", labels, index=1, key="cmp_a")
            b_label = st.selectbox("Newer run (B)", labels, index=0, key="cmp_b")
            if a_label == b_label:
                st.info("Pick two different runs.")
            else:
                comparison = store.compare(by_label[a_label], by_label[b_label])
                if comparison.run_a["filename"] != comparison.run_b["filename"]:
                    st.caption("These runs come from different files, so some KPIs may not line up.")
                st.markdown("**KPIs**")
                st.dataframe(comparison.kpis)
                st.markdown("**Findings**")
                left, middle, right = st.columns(3)
                left.markdown("Only in A (gone in B)")
                left.write(comparison.only_in_a or ["none"])
                middle.markdown("In both")
                middle.write(comparison.in_both or ["none"])
                right.markdown("Only in B (new)")
                right.write(comparison.only_in_b or ["none"])

    with tab_trend:
        filename = st.selectbox("File", store.filenames(), key="hist_file")
        names = store.kpi_names(filename)
        if not names:
            st.info("This file has no numeric KPIs saved yet.")
        else:
            kpi_name = st.selectbox("KPI", names, key="hist_kpi")
            history = store.kpi_history(filename, kpi_name)
            if len(history) < 2:
                st.info("Save at least two unfiltered analyses of this file to see a line.")
            else:
                fig = px.line(
                    history, x="created_at", y="value", markers=True,
                    template="plotly_white", title=f"{kpi_name} across saved analyses of {filename}",
                )
                fig.update_layout(xaxis_title="Saved at", yaxis_title=kpi_name)
                st.plotly_chart(fig, key="hist_chart")

    with tab_manage:
        target = st.selectbox("Run to delete", labels, key="hist_delete")
        sure = st.checkbox("I understand this cannot be undone", key="hist_sure")
        if st.button("Delete run", key="hist_delete_btn", disabled=not sure):
            store.delete_run(by_label[target])
            st.rerun()