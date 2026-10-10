"""Automated exploratory data analysis (EDA)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analysis.metric_selectors import is_additive
from src.profiling.type_detector import (
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    ColumnProfile,
    columns_of_type,
)


@dataclass
class CorrelationPair:
    column_a: str
    column_b: str
    correlation: float
    direction: str  # "positive" or "negative"
    strength: str   # "strong" or "very strong"


@dataclass
class EDAResult:
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    numeric_summary: pd.DataFrame
    categorical_summaries: dict[str, pd.DataFrame]
    correlation_matrix: pd.DataFrame | None
    strong_correlations: list[CorrelationPair] = field(default_factory=list)
    datetime_info: list[dict] = field(default_factory=list)


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    data = df[cols].astype("float64")
    desc = data.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    desc["missing"] = data.isna().sum()
    desc["skew"] = data.skew()
    ordered = ["count", "missing", "mean", "std", "min", "q1", "median", "q3", "max", "skew"]
    return desc[ordered].round(3).rename_axis("column").reset_index()


def categorical_summaries(df: pd.DataFrame, cols: list[str], top_n: int = 5) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for col in cols:
        counts = df[col].value_counts(dropna=True)
        total = int(counts.sum())
        if total == 0:
            continue
        top = counts.head(top_n)
        out[col] = pd.DataFrame(
            {
                "value": top.index.astype(str),
                "count": top.values,
                "pct": (top.values / total * 100).round(1),
            }
        )
    return out


def correlation_analysis(
    df: pd.DataFrame, numeric_cols: list[str], threshold: float = 0.6
) -> tuple[pd.DataFrame | None, list[CorrelationPair]]:
    if len(numeric_cols) < 2:
        return None, []
    corr = df[numeric_cols].astype("float64").corr()
    pairs: list[CorrelationPair] = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append(
                    CorrelationPair(
                        column_a=a,
                        column_b=b,
                        correlation=round(float(r), 3),
                        direction="positive" if r > 0 else "negative",
                        strength="very strong" if abs(r) >= 0.8 else "strong",
                    )
                )
    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return corr, pairs


def describe_gap(days: float) -> str:
    """Turn a typical gap between dates into a human label."""
    if days < 0.9:
        return "intraday (several per day)"
    if days <= 1.5:
        return "daily"
    if 6 <= days <= 8:
        return "weekly"
    if 27 <= days <= 32:
        return "monthly"
    if 85 <= days <= 95:
        return "quarterly"
    if 360 <= days <= 370:
        return "yearly"
    return f"every {days:.1f} days"


def datetime_summary(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    info: list[dict] = []
    for col in cols:
        dates = df[col].dropna().drop_duplicates().sort_values()
        if dates.empty:
            continue
        if len(dates) > 1:
            gap_days = dates.diff().dropna().dt.total_seconds().median() / 86400
            frequency = describe_gap(float(gap_days))
        else:
            frequency = "single date"
        info.append(
            {
                "column": col,
                "start": dates.iloc[0].strftime("%Y-%m-%d"),
                "end": dates.iloc[-1].strftime("%Y-%m-%d"),
                "span_days": int((dates.iloc[-1] - dates.iloc[0]).days),
                "unique_dates": int(len(dates)),
                "frequency": frequency,
            }
        )
    return info


def aggregate_over_time(
    df: pd.DataFrame, date_col: str, metric: str, agg: str | None = None
) -> tuple[pd.DataFrame, str]:
    """Aggregate a metric by day / week / month, choosing the period from the date span.

    Returns (DataFrame with columns 'date' and 'value', period label).
    Additive metrics (revenue, units) are summed; others (price, rating) are averaged.
    """
    data = df[[date_col, metric]].dropna().sort_values(date_col)
    if data.empty:
        return pd.DataFrame(columns=["date", "value"]), "day"
    span_days = (data[date_col].max() - data[date_col].min()).days
    if span_days <= 31:
        rule, label = "D", "day"
    elif span_days <= 180:
        rule, label = "W", "week"
    else:
        rule, label = "MS", "month"
    agg = agg or ("sum" if is_additive(metric) else "mean")
    series = data.set_index(date_col)[metric].astype("float64").resample(rule).agg(agg)
    out = series.reset_index()
    out.columns = ["date", "value"]
    return out, label


def run_eda(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    top_n: int = 5,
    corr_threshold: float = 0.6,
) -> EDAResult:
    numeric_cols = columns_of_type(profiles, NUMERIC)
    categorical_cols = columns_of_type(profiles, CATEGORICAL)
    datetime_cols = columns_of_type(profiles, DATETIME)
    corr, pairs = correlation_analysis(df, numeric_cols, corr_threshold)
    return EDAResult(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        numeric_summary=numeric_summary(df, numeric_cols),
        categorical_summaries=categorical_summaries(df, categorical_cols, top_n),
        correlation_matrix=corr,
        strong_correlations=pairs,
        datetime_info=datetime_summary(df, datetime_cols),
    )