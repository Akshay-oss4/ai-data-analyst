"""Run the whole pipeline (load, validate, clean, analyze, insights) without Streamlit.

Used by the end-to-end tests and the benchmark script.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.ai.insights import InsightsResult, generate_insights
from src.analysis.pipeline import AnalysisBundle, run_analysis
from src.cleaning.cleaner import CleaningOptions, CleaningResult, clean_dataframe
from src.ingestion.loader import LoadResult, load_file
from src.ingestion.validator import ValidationReport, validate_dataframe
from src.profiling.type_detector import ColumnProfile, profile_columns


class AnalysisError(Exception):
    """The file cannot be analyzed. The message is safe to show to the user."""


@dataclass
class EndToEndResult:
    load: LoadResult
    report: ValidationReport
    raw_profiles: list[ColumnProfile]
    cleaning: CleaningResult
    bundle: AnalysisBundle
    insights: InsightsResult
    timings: dict[str, float] = field(default_factory=dict)


def analyze_file(
    data: bytes,
    filename: str,
    sheet_name: str | None = None,
    options: CleaningOptions | None = None,
) -> EndToEndResult:
    """Analyze raw file bytes. Raises FileLoadError or AnalysisError with a friendly message."""
    timings: dict[str, float] = {}

    def timed(stage, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        timings[stage] = round(time.perf_counter() - start, 3)
        return result

    load = timed("load", load_file, data, filename, sheet_name)
    report = timed("validate", validate_dataframe, load.df)
    if report.errors:
        raise AnalysisError(" ".join(report.errors))
    raw_profiles = timed("profile", profile_columns, load.df)
    cleaning = timed("clean", clean_dataframe, load.df, options)
    if cleaning.after_shape[0] == 0 or cleaning.after_shape[1] == 0:
        raise AnalysisError("Nothing is left to analyze after cleaning: every row or column was empty.")
    bundle = timed("analyze", run_analysis, cleaning)
    insights = timed("insights", generate_insights, bundle, cleaning, filename, use_llm=False)
    return EndToEndResult(load, report, raw_profiles, cleaning, bundle, insights, timings)