"""Pure helpers for the dashboard (no Streamlit), so they can be unit-tested."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.kpis import KPI
from src.analysis.metric_selectors import is_additive
from src.analysis.trends import TrendResult

SEVERITY_ICONS = {"high": "🔴", "medium": "🟠", "low": "🟡", "info": "🔵"}
PRIORITY_ICONS = {"high": "🔴", "medium": "🟠", "low": "🟢"}


def kpi_delta(kpi: KPI, trends: list[TrendResult]) -> tuple[str | None, str, str]:
    """Return (delta_text, delta_color, extra_help) for a KPI card.

    The arrow shows how the latest complete period compares with the one before it.
    It is shown for totals, and for averages of non-additive metrics (prices, ratings),
    because those are the cases where the trend is calculated the same way as the KPI.
    """
    if not kpi.column:
        return None, "normal", ""
    trend = next((t for t in trends if t.metric == kpi.column), None)
    if trend is None or trend.last_period_change_pct is None or trend.assessment == "insufficient data":
        return None, "normal", ""

    is_total = kpi.kind == "total"
    is_plain_average = kpi.kind == "average" and not is_additive(kpi.column)
    if not (is_total or is_plain_average):
        return None, "normal", ""

    text = f"{trend.last_period_change_pct:+.1f}% vs previous {trend.period}"
    color = "inverse" if trend.lower_is_better else "normal"  # for cost, down is good
    return text, color, f"Trend: {trend.assessment}."


def _kind(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "number"
    if isinstance(value, str):
        return "text"
    return type(value).__name__


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df where columns that mix numbers, text or dates are converted to text.

    Streamlit shows tables through Arrow, which needs one kind of value per column.
    """
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if series.dtype == object and len({_kind(v) for v in series.dropna()}) > 1:
            out[col] = series.map(lambda v: v if pd.isna(v) else str(v))
    return out