"""Automatic data cleaning. Every change is recorded in a log.

Pipeline:
  names -> trim text & missing tokens -> drop empty -> convert types
  -> remove duplicates -> drop sparse columns -> fill missing values
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.profiling.type_detector import (
    BOOL_MAP,
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    NUMERIC,
    ColumnProfile,
    is_text_dtype,
    parse_datetime_text,
    parse_numeric_text,
    profile_columns,
)

NA_TOKENS = {"", "na", "n/a", "#n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing"}


@dataclass
class CleaningOptions:
    drop_duplicates: bool = True
    numeric_fill: str = "median"        # median | mean | zero | none
    categorical_fill: str = "unknown"   # unknown | mode | none
    drop_column_missing_pct: float = 60.0


@dataclass
class CleaningAction:
    step: str
    column: str | None
    detail: str
    count: int


@dataclass
class CleaningResult:
    df: pd.DataFrame
    log: list[CleaningAction]
    profiles: list[ColumnProfile]
    before_shape: tuple[int, int]
    after_shape: tuple[int, int]


# ---------------------------------------------------------------- steps
def _standardize_column_names(df: pd.DataFrame, log: list[CleaningAction]) -> pd.DataFrame:
    seen: dict[str, int] = {}
    new_names: list[str] = []
    changed = 0
    for col in df.columns:
        name = str(col).strip() or "unnamed"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        if name != str(col):
            changed += 1
        new_names.append(name)
    df.columns = new_names
    if changed:
        log.append(
            CleaningAction("rename_columns", None, "Trimmed spaces / made duplicate column names unique", changed)
        )
    return df


def _normalize_text_and_missing_tokens(df: pd.DataFrame, log: list[CleaningAction]) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if not is_text_dtype(s):
            continue
        mask = s.notna()
        if not mask.any():
            continue
        values = s[mask]
        if not values.map(lambda v: isinstance(v, str)).all():
            continue  # mixed content: leave untouched
        stripped = values.str.strip()
        trimmed = int((stripped != values).sum())
        is_na_token = stripped.str.lower().isin(NA_TOKENS)
        tokens = int(is_na_token.sum())
        if trimmed == 0 and tokens == 0:
            continue
        new = s.copy()
        new.loc[mask] = stripped.where(~is_na_token, np.nan)
        df[col] = new
        if trimmed:
            log.append(CleaningAction("trim_whitespace", col, "Removed leading/trailing spaces", trimmed))
        if tokens:
            log.append(
                CleaningAction("standardize_missing", col, "Blank / N/A / null / '-' entries converted to missing", tokens)
            )
    return df


def _drop_empty(df: pd.DataFrame, log: list[CleaningAction]) -> pd.DataFrame:
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        log.append(CleaningAction("drop_empty_columns", ", ".join(empty_cols), "Column had no values at all", len(empty_cols)))
    if len(df.columns) > 1:
        empty_rows = df.isna().all(axis=1)
        n = int(empty_rows.sum())
        if n:
            df = df.loc[~empty_rows].reset_index(drop=True)
            log.append(CleaningAction("drop_empty_rows", None, "Rows with no values at all were removed", n))
    return df


def _convert_types(df: pd.DataFrame, log: list[CleaningAction]) -> pd.DataFrame:
    profiles = {p.name: p for p in profile_columns(df)}
    for col in df.columns:
        p = profiles[col]
        if not p.stored_as_text:
            continue
        before_missing = int(df[col].isna().sum())

        if p.inferred_type == NUMERIC:
            parsed = parse_numeric_text(df[col])
            failed = int(parsed.isna().sum()) - before_missing
            detail = "Text converted to numbers (currency symbols, % and thousands commas removed)"
            if failed > 0:
                detail += f"; {failed} unreadable value(s) set to missing"
            df[col] = parsed
            log.append(CleaningAction("convert_type", col, detail, int(parsed.notna().sum())))

        elif p.inferred_type == DATETIME:
            parsed = parse_datetime_text(df[col])
            failed = int(parsed.isna().sum()) - before_missing
            detail = "Text converted to dates"
            if failed > 0:
                detail += f"; {failed} unreadable value(s) set to missing"
            df[col] = parsed
            log.append(CleaningAction("convert_type", col, detail, int(parsed.notna().sum())))

        elif p.inferred_type == BOOLEAN:
            mapped = df[col].astype(str).str.strip().str.lower().map(BOOL_MAP)
            df[col] = mapped.astype("boolean")
            log.append(CleaningAction("convert_type", col, "Text (yes/no, true/false, 1/0) converted to True/False", int(df[col].notna().sum())))
    return df


def _handle_duplicates(df: pd.DataFrame, options: CleaningOptions, log: list[CleaningAction]) -> pd.DataFrame:
    n = int(df.duplicated().sum())
    if n and options.drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(CleaningAction("remove_duplicates", None, "Removed exact duplicate rows (kept the first one)", n))
    return df


def _drop_sparse_columns(df: pd.DataFrame, options: CleaningOptions, log: list[CleaningAction]) -> pd.DataFrame:
    if len(df) == 0:
        return df
    pct = df.isna().mean() * 100
    to_drop = pct[pct > options.drop_column_missing_pct].index.tolist()
    for col in to_drop:
        log.append(
            CleaningAction(
                "drop_column", col,
                f"{pct[col]:.1f}% of values were missing (limit {options.drop_column_missing_pct:.0f}%)",
                int(df[col].isna().sum()),
            )
        )
    return df.drop(columns=to_drop) if to_drop else df


def _impute_missing(df: pd.DataFrame, options: CleaningOptions, log: list[CleaningAction]) -> pd.DataFrame:
    for p in profile_columns(df):
        if p.missing_count == 0:
            continue
        col = p.name
        s = df[col]

        if p.inferred_type == NUMERIC:
            if options.numeric_fill == "none":
                log.append(CleaningAction("keep_missing", col, "Missing numbers left as they are (by option)", p.missing_count))
                continue
            if options.numeric_fill == "mean":
                value, label = s.mean(), "mean"
            elif options.numeric_fill == "zero":
                value, label = 0.0, "zero"
            else:
                value, label = s.median(), "median"
            if pd.isna(value):
                continue
            df[col] = s.fillna(value)
            log.append(CleaningAction("fill_missing", col, f"Filled with {label} ({value:.4g})", p.missing_count))

        elif p.inferred_type in (CATEGORICAL, BOOLEAN):
            if options.categorical_fill == "none":
                log.append(CleaningAction("keep_missing", col, "Missing values left as they are (by option)", p.missing_count))
                continue
            if options.categorical_fill == "unknown" and p.inferred_type == CATEGORICAL:
                df[col] = s.astype(object).fillna("Unknown")
                log.append(CleaningAction("fill_missing", col, "Filled with 'Unknown'", p.missing_count))
            else:
                modes = s.mode(dropna=True)
                if modes.empty:
                    continue
                value = modes.iloc[0]
                df[col] = s.fillna(value)
                log.append(CleaningAction("fill_missing", col, f"Filled with the most common value ({value})", p.missing_count))

        else:
            log.append(
                CleaningAction("keep_missing", col, f"{p.inferred_type} column: missing values left as they are", p.missing_count)
            )
    return df


# ---------------------------------------------------------------- public API
def clean_dataframe(df: pd.DataFrame, options: CleaningOptions | None = None) -> CleaningResult:
    """Clean a DataFrame. The input is never modified."""
    options = options or CleaningOptions()
    log: list[CleaningAction] = []
    before_shape = df.shape
    df = df.copy()

    df = _standardize_column_names(df, log)
    df = _normalize_text_and_missing_tokens(df, log)
    df = _drop_empty(df, log)
    df = _convert_types(df, log)
    df = _handle_duplicates(df, options, log)
    df = _drop_sparse_columns(df, options, log)
    df = _impute_missing(df, options, log)

    return CleaningResult(
        df=df,
        log=log,
        profiles=profile_columns(df),
        before_shape=before_shape,
        after_shape=df.shape,
    )


def log_to_dataframe(log: list[CleaningAction]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": a.step.replace("_", " ").capitalize(),
                "column": a.column or "",
                "what happened": a.detail,
                "count": a.count,
            }
            for a in log
        ]
    )