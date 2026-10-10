"""The four dashboard pages. Each function draws one page from an AppContext."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.ai import llm
from src.ai.insights import generate_insights
from src.ai.ollama_client import list_models
from src.analysis.charts import (
    anomaly_chart,
    category_bar_chart,
    correlation_heatmap,
    histogram_chart,
    missing_values_chart,
    time_series_chart,
    trend_chart,
)
from src.analysis.eda import aggregate_over_time
from src.analysis.metric_selectors import pick_primary_date_column, pick_primary_metrics
from src.analysis.pipeline import AnalysisBundle
from src.cleaning.cleaner import CleaningOptions, CleaningResult, log_to_dataframe
from src.config import settings
from src.ingestion.loader import LoadResult
from src.ingestion.validator import ValidationReport
from src.profiling.type_detector import ColumnProfile, profiles_to_dataframe
from src.ui.presenters import arrow_safe
from src.ui.components import finding_list, kpi_cards, render_insights


@dataclass
class AppContext:
    filename: str
    sheet: str | None
    options: CleaningOptions
    load: LoadResult
    report: ValidationReport
    raw_profiles: list[ColumnProfile]
    full_cleaning: CleaningResult      # all rows
    cleaning: CleaningResult           # rows after the sidebar filters
    bundle: AnalysisBundle             # analysis of the filtered rows
    filters_applied: list[str]
    scope_label: str                   # file name + active filters (sent to the AI)
    insights_key: str


# ---------------------------------------------------------------- welcome
def render_welcome() -> None:
    st.title("📊 AI-Powered Automated Data Analyst")
    st.markdown(
        "Upload a **CSV or Excel** file in the sidebar and the app will automatically:\n\n"
        "1. validate and clean the data\n"
        "2. detect column types, KPIs, trends and unusual values\n"
        "3. explain the important findings in plain language\n\n"
        "No file handy? Try the samples in the `data/sample` folder of the project."
    )


# ---------------------------------------------------------------- overview
def _render_main_trend(ctx: AppContext) -> None:
    bundle = ctx.bundle
    trend = next((t for t in bundle.trends if "fitted" in t.series.columns), None)
    if trend is not None:
        marks = [a for a in bundle.anomalies.periods if a.metric == trend.metric]
        st.plotly_chart(trend_chart(trend, marks), key="overview_trend")
        st.caption(f"{trend.metric} is {trend.assessment}.")
        return

    df = ctx.cleaning.df
    date_col = pick_primary_date_column(ctx.cleaning.profiles)
    metrics = pick_primary_metrics(ctx.cleaning.profiles, k=1)
    if date_col and metrics:
        ts, label = aggregate_over_time(df, date_col, metrics[0])
        if not ts.empty:
            st.plotly_chart(time_series_chart(ts, metrics[0], label), key="overview_ts")
            return
    if bundle.eda.numeric_columns:
        st.plotly_chart(histogram_chart(df, bundle.eda.numeric_columns[0]), key="overview_hist")
        return
    st.info("There is no numeric data to chart.")


def render_overview(ctx: AppContext) -> None:
    bundle = ctx.bundle
    rows, cols = ctx.cleaning.after_shape
    st.header("Overview")
    st.caption(f"{ctx.filename} · {rows:,} rows × {cols} columns")
    if ctx.filters_applied:
        st.info("Filters active: " + "; ".join(ctx.filters_applied))

    kpi_cards(bundle.kpis[:8], bundle.trends)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Main trend")
        _render_main_trend(ctx)
    with right:
        st.subheader("Top findings")
        if bundle.findings:
            finding_list(bundle.findings, limit=5)
        else:
            st.info("Nothing notable was found.")
        st.caption("All findings are on the Insights page.")

    st.subheader("AI summary")
    stored = st.session_state.get("insights")
    if stored and stored[0] == ctx.insights_key:
        st.info(stored[1].summary)
    else:
        st.caption("No AI summary yet. Open the Insights page and click Generate AI insights.")


# ---------------------------------------------------------------- data
def render_data(ctx: AppContext) -> None:
    st.header("Data")
    info = ctx.report.info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (original)", f"{info['rows']:,}")
    c2.metric("Columns (original)", info["columns"])
    c3.metric("File size", f"{ctx.load.size_bytes / 1024:,.0f} KB")
    c4.metric("Missing cells", f"{info['missing_pct']}%")

    details = [f"**Type:** {ctx.load.file_type}"]
    if ctx.load.encoding:
        details.append(f"**Encoding:** {ctx.load.encoding}")
    if ctx.load.delimiter:
        names = {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}
        details.append(f"**Delimiter:** {names.get(ctx.load.delimiter, ctx.load.delimiter)}")
    if ctx.load.sheet_name:
        details.append(f"**Sheet:** {ctx.load.sheet_name}")
    st.markdown("  |  ".join(details))
    for note in ctx.load.notes:
        st.warning(note)

    st.success("The file passed validation and can be analyzed.")
    if ctx.report.warnings:
        with st.expander(f"{len(ctx.report.warnings)} validation warning(s)"):
            for warning in ctx.report.warnings:
                st.warning(warning)

    tab_raw, tab_types, tab_log, tab_clean = st.tabs(
        ["Raw preview", "Detected column types", "Cleaning report", "Cleaned data"]
    )
    with tab_raw:
        st.dataframe(arrow_safe(ctx.load.df.head(100)))

    with tab_types:
        st.caption("Types detected on the raw file. 'stored as text' means the values were converted.")
        st.dataframe(profiles_to_dataframe(ctx.raw_profiles))

    with tab_log:
        full = ctx.full_cleaning
        raw_missing = int(ctx.load.df.isna().sum().sum())
        clean_missing = int(full.df.isna().sum().sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{full.after_shape[0]:,}", f"{full.after_shape[0] - full.before_shape[0]:,}")
        m2.metric("Columns", full.after_shape[1], full.after_shape[1] - full.before_shape[1])
        m3.metric("Missing cells", f"{clean_missing:,}", f"{clean_missing - raw_missing:,}", delta_color="inverse")
        if full.log:
            st.dataframe(log_to_dataframe(full.log))
        else:
            st.info("Nothing needed cleaning. The data was already tidy.")
        st.caption("The cleaning report always describes the whole file, not the filtered view.")

    with tab_clean:
        if ctx.filters_applied:
            st.info("Showing the filtered view: " + "; ".join(ctx.filters_applied))
        st.dataframe(profiles_to_dataframe(ctx.cleaning.profiles))
        st.caption("Cleaned data (first 100 rows):")
        st.dataframe(arrow_safe(ctx.cleaning.df.head(100)))
        st.download_button(
            "Download this view as CSV",
            ctx.cleaning.df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------- analysis
def render_analysis(ctx: AppContext) -> None:
    st.header("Analysis")
    if ctx.filters_applied:
        st.info("Filters active: " + "; ".join(ctx.filters_applied))

    bundle, cleaning = ctx.bundle, ctx.cleaning
    eda, cleaned = bundle.eda, cleaning.df

    tabs = st.tabs(
        ["Distributions", "Categories", "Correlations", "Over time", "Trends", "Anomalies", "Data quality"]
    )
    t_dist, t_cat, t_corr, t_time, t_trend, t_anom, t_quality = tabs

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
                "Date column", eda.datetime_columns,
                index=eda.datetime_columns.index(default_date), key="ts_date",
            )
            default_metric = pick_primary_metrics(cleaning.profiles, k=1)[0]
            metric = st.selectbox(
                "Metric", eda.numeric_columns,
                index=eda.numeric_columns.index(default_metric), key="ts_metric",
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

    with t_trend:
        if not bundle.trends:
            st.info("Trend analysis needs a date column and at least one numeric metric.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "metric": t.metric,
                            "assessment": t.assessment,
                            "change along trend (%)": t.change_pct,
                            "latest period vs previous (%)": t.last_period_change_pct,
                            "complete periods": t.n_periods,
                            "per": t.period,
                            "aggregation": t.aggregation,
                            "p-value": t.p_value,
                        }
                        for t in bundle.trends
                    ]
                )
            )
            st.caption(
                "A trend counts only if it is statistically significant (p < 0.05) and at least 5% along the "
                "trend line. Incomplete first/last periods are excluded."
            )
            chosen = st.selectbox("Show chart for", [t.metric for t in bundle.trends], key="trend_metric")
            trend = next(t for t in bundle.trends if t.metric == chosen)
            if "fitted" not in trend.series.columns:
                st.info("Not enough complete periods for a trend (at least 4 are needed).")
            else:
                marks = [a for a in bundle.anomalies.periods if a.metric == chosen]
                st.plotly_chart(trend_chart(trend, marks), key="trend_chart")

    with t_anom:
        an = bundle.anomalies
        if not an.columns and an.rows is None and not an.periods:
            st.success("No unusual values were detected.")
        if an.columns:
            st.markdown("**Unusual values per column** (IQR rule, plus Z-score for symmetric columns)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "column": a.column,
                            "severity": a.severity,
                            "unusual values": a.count,
                            "% of rows": a.pct,
                            "lower limit": a.lower_bound,
                            "upper limit": a.upper_bound,
                            "extreme": a.extreme_count,
                            "methods": ", ".join(a.methods),
                        }
                        for a in an.columns
                    ]
                )
            )
            pick = st.selectbox("Show chart for", [a.column for a in an.columns], key="anom_col")
            ca = next(a for a in an.columns if a.column == pick)
            st.plotly_chart(
                anomaly_chart(
                    cleaned, ca.column, ca.flagged_index, ca.lower_bound, ca.upper_bound,
                    pick_primary_date_column(cleaning.profiles),
                ),
                key="anom_chart",
            )
            st.caption("Most extreme values:")
            st.dataframe(ca.examples)
        if an.rows is not None:
            st.markdown(
                f"**Unusual rows across several columns** ({an.rows.method}): {an.rows.count} rows "
                f"({an.rows.pct}%), using {', '.join(an.rows.columns_used)}"
            )
            st.dataframe(arrow_safe(an.rows.rows.head(50)))
        if an.periods:
            st.markdown("**Unusual periods** (far from the metric's own trend line)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "metric": a.metric,
                            "period": a.period,
                            "value": round(a.value, 2),
                            "expected": round(a.expected, 2),
                            "deviation %": a.deviation_pct,
                            "z-score": a.z_score,
                            "direction": a.direction,
                            "severity": a.severity,
                        }
                        for a in an.periods
                    ]
                )
            )

    with t_quality:
        fig = missing_values_chart(ctx.raw_profiles)
        if fig is None:
            st.success("No missing values were found in the original file.")
        else:
            st.plotly_chart(fig, key="missing_chart")
        st.caption("Based on the raw file, before cleaning and filtering.")


# ---------------------------------------------------------------- insights
def render_insights_page(ctx: AppContext) -> None:
    st.header("Insights")
    if ctx.filters_applied:
        st.info("Filters active: " + "; ".join(ctx.filters_applied) + ". Findings and AI text use this view.")

    st.subheader("Important findings")
    if ctx.bundle.findings:
        finding_list(ctx.bundle.findings)
    else:
        st.info("Nothing notable was found.")

    st.subheader("AI insights and recommendations")
    st.caption(
        "The AI only explains the numbers computed above. With Ollama (the default) everything stays on your computer."
    )

    model_choice = None
    if llm.active_provider() == "ollama":
        available = list_models() or []
        if available:
            default_index = available.index(settings.ollama_model) if settings.ollama_model in available else 0
            model_choice = st.selectbox("Local model", available, index=default_index, key="ai_model")
        else:
            st.info("No local model found, so the built-in rules will be used. To enable the AI: ollama pull llama3.2")

    if st.button("Generate AI insights", key="gen_ai"):
        with st.spinner("Writing insights. A local model can take 30 to 120 seconds on a laptop..."):
            st.session_state["insights"] = (
                ctx.insights_key,
                generate_insights(ctx.bundle, ctx.cleaning, ctx.scope_label, model_choice),
            )

    stored = st.session_state.get("insights")
    if stored and stored[0] == ctx.insights_key:
        render_insights(stored[1])