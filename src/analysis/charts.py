"""Plotly chart builders. Each function returns a plotly Figure (or None if there is nothing to draw)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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