"""Trend detection: is each metric improving, declining or stable over time?

Method: aggregate the metric by day/week/month, drop incomplete periods, fit a straight
line, and call the trend real only if it is statistically significant (p < 0.05) AND
large enough to matter (at least 5% change along the line).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.eda import aggregate_over_time
from src.analysis.metric_selectors import (
    is_additive,
    pick_primary_date_column,
    pick_primary_metrics,
)
from src.profiling.type_detector import ColumnProfile

MIN_PERIODS = 4
SIGNIFICANCE = 0.05
MIN_CHANGE_PCT = 5.0

# For these metrics, going DOWN is good news
_LOWER_IS_BETTER_RE = re.compile(
    r"cost|expense|churn|complaint|returns|returned|refund|defect|error|loss|delay|late|"
    r"cancel|bounce|wait|downtime",
    re.IGNORECASE,
)


def lower_is_better(name: str) -> bool:
    return bool(_LOWER_IS_BETTER_RE.search(str(name)))


@dataclass
class TrendResult:
    metric: str
    date_column: str
    period: str                       # day | week | month
    aggregation: str                  # sum | mean
    n_periods: int                    # complete periods used
    trend: str                        # up | down | flat | unknown
    assessment: str                   # improving | declining | stable | insufficient data
    change_pct: float | None          # change along the trend line, first to last period
    last_period_change_pct: float | None
    p_value: float | None
    r_squared: float | None
    lower_is_better: bool
    series: pd.DataFrame = field(default_factory=pd.DataFrame)  # date, value, fitted


def drop_incomplete_periods(
    ts: pd.DataFrame, label: str, data_min: pd.Timestamp, data_max: pd.Timestamp
) -> pd.DataFrame:
    """Remove a partial first/last period so it is not mistaken for a real drop."""
    if label == "day" or len(ts) < 2:
        return ts
    keep = [True] * len(ts)
    first, last = ts["date"].iloc[0], ts["date"].iloc[-1]
    dmin = pd.Timestamp(data_min).normalize()
    dmax = pd.Timestamp(data_max).normalize()
    if label == "month":  # labels are month starts
        if dmin > first:
            keep[0] = False
        if dmax < last + pd.offsets.MonthEnd(0):
            keep[-1] = False
    elif label == "week":  # labels are week-ending Sundays
        if dmin > first - pd.Timedelta(days=6):
            keep[0] = False
        if dmax < last:
            keep[-1] = False
    return ts.loc[keep].reset_index(drop=True)


def analyze_trend(
    df: pd.DataFrame, date_col: str, metric: str, agg: str | None = None
) -> TrendResult:
    agg_used = agg or ("sum" if is_additive(metric) else "mean")
    lib = lower_is_better(metric)

    def unavailable(period: str, n: int, series: pd.DataFrame) -> TrendResult:
        return TrendResult(
            metric=metric, date_column=date_col, period=period, aggregation=agg_used,
            n_periods=n, trend="unknown", assessment="insufficient data",
            change_pct=None, last_period_change_pct=None, p_value=None, r_squared=None,
            lower_is_better=lib, series=series,
        )

    data = df[[date_col, metric]].dropna()
    ts, label = aggregate_over_time(df, date_col, metric, agg)
    if ts.empty:
        return unavailable(label, 0, ts)

    ts = drop_incomplete_periods(ts, label, data[date_col].min(), data[date_col].max())
    ts = ts.dropna(subset=["value"]).reset_index(drop=True)
    n = len(ts)
    if n < MIN_PERIODS:
        return unavailable(label, n, ts)

    x = np.arange(n)
    y = ts["value"].to_numpy(dtype=float)
    fit = stats.linregress(x, y)
    p_value = float(fit.pvalue) if np.isfinite(fit.pvalue) else 1.0
    r_squared = float(fit.rvalue**2) if np.isfinite(fit.rvalue) else 0.0
    ts["fitted"] = fit.intercept + fit.slope * x

    start, end = float(ts["fitted"].iloc[0]), float(ts["fitted"].iloc[-1])
    mean_level = abs(float(y.mean()))
    # Use the starting level as the base, unless it is tiny compared with the average
    denom = abs(start) if abs(start) >= 0.1 * mean_level and abs(start) > 0 else mean_level
    change_pct = (end - start) / denom * 100 if denom > 0 else 0.0

    if p_value < SIGNIFICANCE and abs(change_pct) >= MIN_CHANGE_PCT:
        trend = "up" if change_pct > 0 else "down"
        is_good = (trend == "down") == lib
        assessment = "improving" if is_good else "declining"
    else:
        trend, assessment = "flat", "stable"

    prev, last = y[-2], y[-1]
    last_change = float((last - prev) / abs(prev) * 100) if prev != 0 else None

    return TrendResult(
        metric=metric, date_column=date_col, period=label, aggregation=agg_used,
        n_periods=n, trend=trend, assessment=assessment,
        change_pct=round(float(change_pct), 2),
        last_period_change_pct=None if last_change is None else round(last_change, 2),
        p_value=round(p_value, 4), r_squared=round(r_squared, 3),
        lower_is_better=lib, series=ts,
    )


def analyze_trends(
    df: pd.DataFrame, profiles: list[ColumnProfile], max_metrics: int = 6
) -> list[TrendResult]:
    date_col = pick_primary_date_column(profiles)
    if date_col is None:
        return []
    return [analyze_trend(df, date_col, m) for m in pick_primary_metrics(profiles, k=max_metrics)]