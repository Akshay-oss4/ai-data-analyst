"""Dashboard filters (pure pandas, no Streamlit): date range and category selection."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from src.profiling.type_detector import ColumnProfile


@dataclass
class FilterSpec:
    date_column: str | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    categories: dict[str, list[str]] = field(default_factory=dict)  # column -> chosen values


def apply_filters(df: pd.DataFrame, spec: FilterSpec) -> tuple[pd.DataFrame, list[str]]:
    """Return (filtered_df, descriptions of the filters that actually removed something).

    A date range that covers the whole data is ignored, so rows with a missing date are kept.
    """
    mask = pd.Series(True, index=df.index)
    applied: list[str] = []

    if (
        spec.date_column
        and spec.date_column in df.columns
        and pd.api.types.is_datetime64_any_dtype(df[spec.date_column])
    ):
        col = df[spec.date_column]
        full_min, full_max = col.min(), col.max()
        start_on = (
            spec.start is not None
            and pd.notna(full_min)
            and spec.start.normalize() > full_min.normalize()
        )
        end_on = (
            spec.end is not None
            and pd.notna(full_max)
            and spec.end.normalize() < full_max.normalize()
        )
        if start_on:
            mask &= col >= spec.start.normalize()
        if end_on:
            mask &= col < spec.end.normalize() + pd.Timedelta(days=1)  # include the whole end day
        if start_on or end_on:
            low = spec.start if start_on else full_min
            high = spec.end if end_on else full_max
            applied.append(f"{spec.date_column}: {low:%Y-%m-%d} to {high:%Y-%m-%d}")

    for column, values in spec.categories.items():
        if values and column in df.columns:
            mask &= df[column].astype(str).isin([str(v) for v in values])
            applied.append(f"{column} in {', '.join(str(v) for v in values)}")

    return df.loc[mask].reset_index(drop=True), applied


def refresh_profiles(profiles: list[ColumnProfile], df: pd.DataFrame) -> list[ColumnProfile]:
    """Keep the detected column types, but recompute counts for the filtered rows."""
    total = len(df)
    refreshed: list[ColumnProfile] = []
    for p in profiles:
        if p.name not in df.columns:
            continue
        s = df[p.name]
        missing, unique = int(s.isna().sum()), int(s.nunique())
        refreshed.append(
            replace(
                p,
                missing_count=missing,
                missing_pct=round(missing / total * 100, 2) if total else 0.0,
                unique_count=unique,
                unique_pct=round(unique / total * 100, 2) if total else 0.0,
                sample_values=[str(v) for v in s.dropna().unique()[:3]],
            )
        )
    return refreshed