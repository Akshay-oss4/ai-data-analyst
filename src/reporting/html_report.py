"""Self-contained HTML report (pure Python). Open in a browser; Ctrl+P gives a PDF.

All text that comes from the user's data is HTML-escaped, so a column called
"<script>" cannot inject code into the report.
"""
from __future__ import annotations

import math
from datetime import datetime
from html import escape as esc

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from src.ai.insights import InsightsResult
from src.analysis.charts import (
    anomaly_chart,
    category_bar_chart,
    correlation_heatmap,
    histogram_chart,
    missing_values_chart,
    trend_chart,
)
from src.analysis.metric_selectors import pick_primary_date_column, pick_primary_metrics
from src.analysis.pipeline import AnalysisBundle
from src.cleaning.cleaner import CleaningResult, log_to_dataframe
from src.profiling.type_detector import ColumnProfile, profiles_to_dataframe
from src.ui.presenters import kpi_delta

_CSS = """
:root { --fg:#111827; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; }
body { font-family: "Segoe UI", Arial, sans-serif; color: var(--fg); max-width: 1000px;
       margin: 24px auto; padding: 0 16px; line-height: 1.5; }
h1 { font-size: 26px; margin-bottom: 4px; }
h2 { border-bottom: 2px solid var(--line); padding-bottom: 6px; margin-top: 36px; }
h3 { margin-top: 22px; }
.meta { color: var(--muted); font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }
.card { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.card .label { color: var(--muted); font-size: 12px; }
.card .value { font-size: 22px; font-weight: 600; }
.delta-good { color: #15803d; font-size: 13px; }
.delta-bad { color: #b91c1c; font-size: 13px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; color: #fff; }
.sev-high { background: #dc2626; } .sev-medium { background: #ea580c; }
.sev-low { background: #ca8a04; } .sev-info { background: #2563eb; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
.table-wrap { overflow-x: auto; }
.callout { background: #eff6ff; border-left: 4px solid var(--accent); padding: 10px 14px; border-radius: 4px; }
.warn { background: #fff7ed; border-left-color: #ea580c; }
.chart { margin: 12px 0; }
pre { background: #f3f4f6; padding: 10px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; }
@media print {
  body { max-width: none; margin: 0; }
  h2 { page-break-after: avoid; }
  .card, .chart, tr { page-break-inside: avoid; }
}
"""


class _Raw(str):
    """Text that is already safe HTML and must not be escaped again."""


def _fmt(value) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ""
        return f"{float(value):.3e}" if abs(value) >= 1e15 else f"{float(value):,.2f}"
    if isinstance(value, pd.Timestamp):
        return f"{value:%Y-%m-%d}"
    return str(value)

def _cell(value) -> str:
    return str(value) if isinstance(value, _Raw) else esc(_fmt(value))


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{esc(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _df_table(df: pd.DataFrame) -> str:
    return _table([str(c) for c in df.columns], df.values.tolist())


def _badge(level: str) -> _Raw:
    level = level if level in {"high", "medium", "low", "info"} else "info"
    return _Raw(f'<span class="badge sev-{level}">{esc(level)}</span>')


def _based_on(ids: list[str]) -> str:
    return f' <span class="meta">(based on {esc(", ".join(ids))})</span>' if ids else ""


class _Charts:
    """Turns Plotly figures into HTML. The Plotly library is embedded once in the page head."""

    def __init__(self, mode: str):
        self.enabled = mode == "inline"

    def html(self, build) -> str:
        if not self.enabled:
            return ""
        try:
            fig = build()
            if fig is None:
                return ""
            fig.update_layout(height=380, margin=dict(l=50, r=20, t=60, b=40))
            div = fig.to_html(
                full_html=False, include_plotlyjs=False,
                default_width="100%", default_height="380px", config={"displaylogo": False},
            )
        except Exception:  # noqa: BLE001 - one broken chart must not break the whole report
            return '<p class="meta">(this chart could not be drawn)</p>'
        return f'<div class="chart">{div}</div>'


# ---------------------------------------------------------------- sections
def _header(title, scope_label, cleaning, filters_applied, generated_at, charts_on) -> str:
    rows, cols = cleaning.after_shape
    html = (
        f"<h1>{esc(title)}</h1>"
        f'<p class="meta">Data: {esc(scope_label)} · {rows:,} rows × {cols} columns · '
        f"Generated {generated_at:%Y-%m-%d %H:%M}</p>"
    )
    if filters_applied:
        html += f'<div class="callout warn">Filtered view: {esc("; ".join(filters_applied))}</div>'
    if not charts_on:
        html += '<p class="meta">Charts were left out of this report.</p>'
    return html


def _summary_section(insights: InsightsResult) -> str:
    source = (
        f"Written by {insights.model}"
        if insights.source == "llm"
        else "Automatic summary (rule-based; no AI model was used)"
    )
    html = (
        "<h2>Executive summary</h2>"
        f'<div class="callout"><p>{esc(insights.summary)}</p><p class="meta">{esc(source)}</p></div>'
    )
    if insights.unsupported_numbers:
        html += (
            '<div class="callout warn">Please double-check these numbers in the AI text. '
            f"They were not found in the computed facts: {esc(', '.join(insights.unsupported_numbers))}</div>"
        )
    return html


def _kpi_section(bundle: AnalysisBundle) -> str:
    cards = []
    for kpi in bundle.kpis:
        delta, color, _ = kpi_delta(kpi, bundle.trends)
        delta_html = ""
        if delta:
            rising = not delta.startswith("-")
            good = rising != (color == "inverse")  # for cost-like metrics, falling is good
            delta_html = (
                f'<div class="{"delta-good" if good else "delta-bad"}">'
                f'{"▲" if rising else "▼"} {esc(delta)}</div>'
            )
        cards.append(
            f'<div class="card"><div class="label">{esc(kpi.name)}</div>'
            f'<div class="value">{esc(kpi.display)}</div>{delta_html}</div>'
        )
    return "<h2>Key metrics</h2>" + f'<div class="cards">{"".join(cards)}</div>'


def _findings_section(bundle: AnalysisBundle) -> str:
    if not bundle.findings:
        return "<h2>Important findings</h2><p>Nothing notable was found.</p>"
    rows = [
        [f"F{i}" if i <= 12 else "", _badge(f.severity), f.title, f.detail]
        for i, f in enumerate(bundle.findings, start=1)
    ]
    return "<h2>Important findings</h2>" + _table(["ID", "Severity", "Finding", "Details"], rows)


def _insights_section(insights: InsightsResult) -> str:
    html = "<h2>Insights and recommendations</h2><h3>Insights</h3><ul>"
    for item in insights.insights:
        html += f"<li><strong>{esc(item.title)}</strong>: {esc(item.explanation)}{_based_on(item.based_on)}</li>"
    html += "</ul><h3>Recommendations</h3>"
    rows = [
        [_badge(r.priority), r.action, r.reason, ", ".join(r.based_on)]
        for r in insights.recommendations
    ]
    html += _table(["Priority", "Action", "Why", "Based on"], rows)
    if insights.caveats:
        html += f'<p class="meta">Caveats: {esc(" ".join(insights.caveats))}</p>'
    return html


def _trends_section(bundle: AnalysisBundle, charts: _Charts) -> str:
    html = "<h2>Trends</h2>"
    if not bundle.trends:
        return html + '<p class="meta">Trend analysis needs a date column and a numeric metric.</p>'
    rows = [
        [
            t.metric, t.assessment,
            "" if t.change_pct is None else f"{t.change_pct:+.1f}%",
            "" if t.last_period_change_pct is None else f"{t.last_period_change_pct:+.1f}%",
            f"{t.n_periods} {t.period}s", t.aggregation,
            "" if t.p_value is None else f"{t.p_value:.4f}",
        ]
        for t in bundle.trends
    ]
    html += _table(
        ["Metric", "Assessment", "Change along trend", "Latest period vs previous", "Complete periods", "Aggregation", "p-value"],
        rows,
    )
    for trend in [t for t in bundle.trends if "fitted" in t.series.columns][:3]:
        marks = [a for a in bundle.anomalies.periods if a.metric == trend.metric]
        html += charts.html(lambda t=trend, m=marks: trend_chart(t, m))
    return html


def _anomalies_section(bundle: AnalysisBundle, cleaning: CleaningResult, charts: _Charts) -> str:
    html = "<h2>Unusual values</h2>"
    an = bundle.anomalies
    if not an.columns and an.rows is None and not an.periods:
        return html + "<p>No unusual values were detected.</p>"

    if an.columns:
        html += "<h3>Per column</h3>" + _table(
            ["Column", "Severity", "Unusual values", "% of rows", "Lower limit", "Upper limit", "Extreme", "Methods"],
            [
                [a.column, _badge(a.severity), a.count, a.pct, a.lower_bound, a.upper_bound, a.extreme_count, ", ".join(a.methods)]
                for a in an.columns
            ],
        )
        date_col = pick_primary_date_column(cleaning.profiles)
        for a in an.columns[:2]:
            html += charts.html(
                lambda a=a: anomaly_chart(cleaning.df, a.column, a.flagged_index, a.lower_bound, a.upper_bound, date_col)
            )

    if an.rows is not None:
        html += (
            f"<h3>Unusual rows across several columns</h3>"
            f"<p>{an.rows.count} rows ({an.rows.pct}%) look unusual when {esc(', '.join(an.rows.columns_used))} "
            f"are considered together ({esc(an.rows.method)}). The most unusual ones:</p>"
        )
        frame = an.rows.rows
        keep = [c for c in an.rows.columns_used + ["anomaly_score", "also_unusual_in"] if c in frame.columns]
        html += _df_table(frame[keep].head(10))

    if an.periods:
        html += "<h3>Unusual periods</h3>" + _table(
            ["Metric", "Period", "Value", "Expected", "Deviation", "Direction", "Severity"],
            [
                [a.metric, a.period, a.value, a.expected, f"{a.deviation_pct:+.1f}%", a.direction, _badge(a.severity)]
                for a in an.periods
            ],
        )
    return html


def _exploration_section(bundle: AnalysisBundle, cleaning: CleaningResult, charts: _Charts) -> str:
    html = "<h2>Distributions and relationships</h2>"
    df, eda = cleaning.df, bundle.eda

    if not eda.numeric_summary.empty:
        html += "<h3>Numeric columns</h3>" + _df_table(eda.numeric_summary)
    for col in pick_primary_metrics(cleaning.profiles, k=3):
        html += charts.html(lambda c=col: histogram_chart(df, c))

    for col in list(eda.categorical_summaries)[:2]:
        html += f"<h3>{esc(col)}</h3>" + _df_table(eda.categorical_summaries[col])
        html += charts.html(lambda c=col: category_bar_chart(df, c))

    if eda.correlation_matrix is not None:
        html += "<h3>Correlations</h3>"
        if eda.strong_correlations:
            html += "<ul>" + "".join(
                f"<li><strong>{esc(p.column_a)}</strong> and <strong>{esc(p.column_b)}</strong>: "
                f"{esc(p.strength)} {esc(p.direction)} relationship (r = {p.correlation})</li>"
                for p in eda.strong_correlations[:5]
            ) + "</ul>"
        else:
            html += "<p>No strong correlations were found.</p>"
        html += '<p class="meta">Correlation shows that two columns move together, not that one causes the other.</p>'
        html += charts.html(lambda: correlation_heatmap(eda.correlation_matrix))
    return html


def _data_section(
    full_cleaning: CleaningResult, cleaning: CleaningResult, raw_profiles: list[ColumnProfile], charts: _Charts
) -> str:
    html = "<h2>Data and cleaning</h2><h3>Cleaning steps applied to the whole file</h3>"
    log = log_to_dataframe(full_cleaning.log)
    html += _df_table(log) if not log.empty else "<p>Nothing needed cleaning.</p>"
    html += "<h3>Column types after cleaning</h3>" + _df_table(profiles_to_dataframe(cleaning.profiles))
    html += charts.html(lambda: missing_values_chart(raw_profiles))
    return html


def _method_section(insights: InsightsResult) -> str:
    html = (
        "<h2>How this report was made</h2><ul>"
        "<li><strong>Cleaning:</strong> spaces trimmed, blank/N-A entries treated as missing, text converted to numbers, "
        "dates and yes/no values, duplicate rows removed, missing values filled (see the cleaning steps above).</li>"
        "<li><strong>Trends:</strong> each metric is added up (or averaged) per day, week or month. Incomplete first and last "
        "periods are left out. A straight line is fitted, and a trend is reported only if it is statistically significant "
        "(p &lt; 0.05) and at least 5% along the line.</li>"
        "<li><strong>Unusual values:</strong> the IQR rule (1.5 times the middle-50% range) on every numeric column, plus "
        "Z-score (more than 3 standard deviations) on roughly symmetric columns. Rows unusual across several columns are "
        "found with a multivariate method, and periods far from their own trend line are flagged.</li>"
        "<li><strong>AI text:</strong> the language model only receives a list of computed facts, and every number it writes "
        "is checked against that list.</li>"
        "<li><strong>Limits:</strong> these are statistical signals, not proof of cause. A person should judge each one.</li></ul>"
    )
    if insights.source == "llm" and insights.fact_sheet:
        html += f"<details><summary>The facts the AI was given</summary><pre>{esc(insights.fact_sheet)}</pre></details>"
    return html


# ---------------------------------------------------------------- public API
def build_html_report(
    *,
    title: str,
    scope_label: str,
    cleaning: CleaningResult,
    full_cleaning: CleaningResult,
    raw_profiles: list[ColumnProfile],
    bundle: AnalysisBundle,
    insights: InsightsResult,
    filters_applied: list[str],
    chart_mode: str = "inline",
    generated_at: datetime | None = None,
) -> str:
    """Build the report. chart_mode is 'inline' (interactive charts, works offline) or 'none'."""
    generated_at = generated_at or datetime.now()
    charts = _Charts(chart_mode)

    body = "".join(
        [
            _header(title, scope_label, cleaning, filters_applied, generated_at, charts.enabled),
            _summary_section(insights),
            _kpi_section(bundle),
            _findings_section(bundle),
            _insights_section(insights),
            _trends_section(bundle, charts),
            _anomalies_section(bundle, cleaning, charts),
            _exploration_section(bundle, cleaning, charts),
            _data_section(full_cleaning, cleaning, raw_profiles, charts),
            _method_section(insights),
        ]
    )
    plotly_script = f"<script>{get_plotlyjs()}</script>" if charts.enabled else ""
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}</style>{plotly_script}</head>"
        f"<body>{body}</body></html>"
    )