"""Shared helper for the end-to-end tests: build every output from one analysis."""
from src.database.history import HistoryStore, build_run_record
from src.reporting.excel_report import build_excel_report
from src.reporting.html_report import build_html_report


def build_outputs(result, filename, tmp_path, chart_mode="none"):
    common = dict(
        title=f"Report: {filename}", scope_label=filename, cleaning=result.cleaning,
        full_cleaning=result.cleaning, bundle=result.bundle, insights=result.insights, filters_applied=[],
    )
    html = build_html_report(**common, raw_profiles=result.raw_profiles, chart_mode=chart_mode)
    xlsx = build_excel_report(**common)
    record = build_run_record(filename, None, filename, [], result.cleaning, result.bundle, result.insights)
    run_id, _ = HistoryStore(tmp_path / "history.db").save_run(record)
    return html, xlsx, run_id