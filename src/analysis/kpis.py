"""Automatic KPI selection and calculation."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analysis.metric_selectors import (
    is_additive,
    pick_primary_date_column,
    pick_primary_metrics,
)
from src.profiling.type_detector import BOOLEAN, CATEGORICAL, ColumnProfile


@dataclass
class KPI:
    name: str
    value: float | int | str
    display: str
    kind: str  # count | period | total | average | rate | top_category
    column: str | None = None
    description: str = ""


def format_number(value: float) -> str:
    """Compact, readable numbers: 1.23M, 12.5K, 1,234, 3.14."""
    if pd.isna(value):
        return "n/a"
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.2f}B"
    if a >= 1e6:
        return f"{value / 1e6:.2f}M"
    if a >= 1e4:
        return f"{value / 1e3:.1f}K"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def compute_kpis(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    max_metrics: int = 4,
    max_booleans: int = 3,
    max_categoricals: int = 2,
) -> list[KPI]:
    kpis: list[KPI] = [
        KPI("Rows analyzed", len(df), f"{len(df):,}", "count", None, "Number of rows after cleaning")
    ]

    date_col = pick_primary_date_column(profiles)
    if date_col:
        dates = df[date_col].dropna()
        if not dates.empty:
            start, end = dates.min(), dates.max()
            span = (end - start).days
            kpis.append(
                KPI(
                    "Period covered",
                    f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
                    f"{span:,} days",
                    "period",
                    date_col,
                    f"{start:%Y-%m-%d} to {end:%Y-%m-%d} (column: {date_col})",
                )
            )

    for col in pick_primary_metrics(profiles, k=max_metrics):
        s = df[col].dropna()
        if s.empty:
            continue
        if is_additive(col):
            total = float(s.sum())
            kpis.append(KPI(f"Total {col}", total, format_number(total), "total", col, f"Sum of {col} over all rows"))
        mean = float(s.mean())
        kpis.append(
            KPI(
                f"Average {col}",
                mean,
                format_number(mean),
                "average",
                col,
                f"Mean of {col} (median: {format_number(float(s.median()))})",
            )
        )

    booleans = [p for p in profiles if p.inferred_type == BOOLEAN][:max_booleans]
    for p in booleans:
        s = df[p.name].dropna()
        if s.empty:
            continue
        rate = float(s.astype("float64").mean() * 100)
        kpis.append(
            KPI(f"{p.name}: % true", rate, f"{rate:.1f}%", "rate", p.name, f"Share of rows where {p.name} is True/1")
        )

    categoricals = [p for p in profiles if p.inferred_type == CATEGORICAL and 2 <= p.unique_count <= 15]
    categoricals.sort(key=lambda p: p.unique_count)
    for p in categoricals[:max_categoricals]:
        shares = df[p.name].value_counts(normalize=True)
        if shares.empty:
            continue
        top, share = shares.index[0], float(shares.iloc[0] * 100)
        kpis.append(
            KPI(
                f"Top {p.name}",
                str(top),
                f"{top} ({share:.0f}%)",
                "top_category",
                p.name,
                f"Most common {p.name} value and its share of rows",
            )
        )
    return kpis
