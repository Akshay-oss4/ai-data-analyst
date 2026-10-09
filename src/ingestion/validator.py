"""Validate a loaded DataFrame and report problems in plain language.

errors   -> the dataset cannot be analyzed
warnings -> the dataset can be analyzed, but the user should know about an issue
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MIN_RECOMMENDED_ROWS = 10
MAX_RECOMMENDED_ROWS = 500_000
MAX_RECOMMENDED_COLUMNS = 200
HIGH_MISSING_PCT = 50.0


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    rows, cols = df.shape

    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum()) if total_cells else 0
    report.info = {
        "rows": rows,
        "columns": cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0.0,
        "duplicate_rows": int(df.duplicated().sum()) if rows else 0,
    }

    # ---- Errors: analysis is impossible ----
    if cols == 0:
        report.errors.append("The file has no columns.")
    if rows == 0:
        report.errors.append("The file has column headers but no data rows.")
    if report.errors:
        return report

    # ---- Warnings: analysis possible, but the user should know ----
    if cols == 1:
        report.warnings.append(
            "Only one column was found. The delimiter may have been detected incorrectly."
        )
    if rows < MIN_RECOMMENDED_ROWS:
        report.warnings.append(
            f"Only {rows} rows found. Statistical results may be unreliable on so little data."
        )
    if rows > MAX_RECOMMENDED_ROWS:
        report.warnings.append(f"{rows:,} rows is large. The app may be slow.")
    if cols > MAX_RECOMMENDED_COLUMNS:
        report.warnings.append(f"{cols} columns is a lot. The analysis may be slow.")

    unnamed = [str(c) for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        report.warnings.append(
            f"{len(unnamed)} column(s) have no header name ({', '.join(unnamed[:5])}). "
            "The header row may not be the first row of the file."
        )

    empty_cols = [str(c) for c in df.columns[df.isna().all()]]
    if empty_cols:
        report.warnings.append(
            f"{len(empty_cols)} column(s) are completely empty: {', '.join(empty_cols[:5])}."
        )

    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        report.warnings.append(f"{empty_rows} row(s) are completely empty.")

    if report.info["duplicate_rows"]:
        report.warnings.append(
            f"{report.info['duplicate_rows']} duplicate row(s) found (handled in the cleaning step)."
        )

    if report.info["missing_pct"] > HIGH_MISSING_PCT:
        report.warnings.append(
            f"{report.info['missing_pct']}% of all cells are missing. Results may be unreliable."
        )

    return report