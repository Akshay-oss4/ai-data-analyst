"""Plotly chart builders. Each function returns a plotly Figure (or None if there is nothing to draw)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis.anomalies import PeriodAnomaly
from src.analysis.trends import TrendResult
from src.profiling.type_detector import ColumnProfile

TEMPLATE = "plotly_white"


def histogram_chart(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, marginal="box", template=TEMPLATE, title=f"Distribution of {col}")
    fig.update_layout(bargap=0.05)
    return fig


def category_bar_chart(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    counts = df[col].value_counts().head(top_n)
    plot_df = pd.DataFrame({"category": counts.index.astype(str), "count": counts.values})
    fig = px.bar(plot_df, x="category", y="count", template=TEMPLATE, title=f"Most common values of {col}")
    fig.update_layout(xaxis_title=col, yaxis_title="Rows")
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    return px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template=TEMPLATE,
        title="Correlation between numeric columns",
    )


def time_series_chart(ts: pd.DataFrame, metric: str, period_label: str) -> go.Figure:
    fig = px.line(
        ts,
        x="date",
        y="value",
        markers=len(ts) <= 60,
        template=TEMPLATE,
        title=f"{metric} by {period_label}",
    )
    fig.update_layout(xaxis_title="", yaxis_title=metric)
    return fig


def missing_values_chart(profiles: list[ColumnProfile]) -> go.Figure | None:
    rows = [(p.name, p.missing_pct, p.missing_count) for p in profiles if p.missing_count > 0]
    if not rows:
        return None
    data = pd.DataFrame(rows, columns=["column", "missing_pct", "missing_count"]).sort_values("missing_pct")
    fig = px.bar(
        data,
        x="missing_pct",
        y="column",
        orientation="h",
        hover_data=["missing_count"],
        template=TEMPLATE,
        title="Missing values by column (original file)",
    )
    fig.update_layout(xaxis_title="% missing", yaxis_title="")
    return fig


def trend_chart(trend: TrendResult, period_anomalies: list[PeriodAnomaly] | None = None) -> go.Figure:
    """Metric by period with its trend line; unusual periods are marked in red."""
    s = trend.series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s["value"], mode="lines+markers", name=trend.metric))
    if "fitted" in s.columns:
        fig.add_trace(
            go.Scatter(x=s["date"], y=s["fitted"], mode="lines", name="Trend line", line=dict(dash="dash"))
        )
    if period_anomalies:
        fig.add_trace(
            go.Scatter(
                x=[a.date for a in period_anomalies],
                y=[a.value for a in period_anomalies],
                mode="markers",
                name="Unusual period",
                marker=dict(color="#dc2626", size=12, symbol="x"),
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title=f"{trend.metric} per {trend.period} ({trend.aggregation}), {trend.assessment}",
        xaxis_title="",
        yaxis_title=trend.metric,
    )
    return fig


def anomaly_chart(
    df: pd.DataFrame,
    col: str,
    flagged_index: list,
    lower: float | None = None,
    upper: float | None = None,
    date_col: str | None = None,
) -> go.Figure:
    """Every value of a column; unusual ones in red, with the normal range drawn as dashed lines."""
    data = df[[col]].copy()
    data["x"] = df[date_col] if date_col else np.arange(len(df))
    data["status"] = np.where(data.index.isin(flagged_index), "Unusual", "Normal")
    if len(data) > 20000:  # keep the chart fast: sample normal points, keep every unusual one
        normal = data[data["status"] == "Normal"].sample(20000, random_state=0)
        data = pd.concat([normal, data[data["status"] == "Unusual"]])
    fig = px.scatter(
        data,
        x="x",
        y=col,
        color="status",
        color_discrete_map={"Normal": "#9ca3af", "Unusual": "#dc2626"},
        opacity=0.7,
        template=TEMPLATE,
        title=f"Unusual values in {col}",
    )
    if lower is not None:
        fig.add_hline(y=lower, line_dash="dash", line_color="#6b7280", annotation_text="lower limit")
    if upper is not None:
        fig.add_hline(y=upper, line_dash="dash", line_color="#6b7280", annotation_text="upper limit")
    fig.update_layout(xaxis_title=date_col or "Row number", yaxis_title=col)
    return fig