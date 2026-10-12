"""Excel workbook export (pure Python, via openpyxl).

Text cells that start with = + - or @ get a leading apostrophe, so Excel can never run
them as formulas ("formula injection" from untrusted data).
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from openpyxl.styles import Font
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from src.ai.insights import InsightsResult
from src.analysis.pipeline import AnalysisBundle
from src.cleaning.cleaner import CleaningResult, log_to_dataframe
from src.profiling.type_detector import profiles_to_dataframe

_FORMULA_STARTS = ("=", "+", "-", "@")


_MAX_CELL_CHARS = 32_000  # Excel allows 32,767 characters per cell

def _safe_text(value):
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)  # control characters make openpyxl raise an error
        if len(value) > _MAX_CELL_CHARS:
            value = value[:_MAX_CELL_CHARS] + "…"
        if value.startswith(_FORMULA_STARTS):
            value = "'" + value
    return value


def sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_safe_text(str(c)) for c in out.columns]
    for col in out.columns:
        series = out[col]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            out[col] = series.dt.tz_localize(None)  # Excel cannot store time zones
        elif series.dtype == object or pd.api.types.is_string_dtype(series):
            out[col] = series.map(_safe_text)
    return out


def _format_workbook(writer) -> None:
    for sheet in writer.book.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for column_cells in sheet.columns:
            longest = max(
                (len(str(c.value)) if c.value is not None else 0 for c in column_cells[:200]),
                default=8,
            )
            sheet.column_dimensions[column_cells[0].column_letter].width = min(max(longest + 2, 8), 60)


def build_excel_report(
    *,
    title: str,
    scope_label: str,
    cleaning: CleaningResult,
    full_cleaning: CleaningResult,
    bundle: AnalysisBundle,
    insights: InsightsResult,
    filters_applied: list[str],
    generated_at: datetime | None = None,
    max_rows: int = 100_000,
) -> bytes:
    generated_at = generated_at or datetime.now()
    rows, cols = cleaning.after_shape

    summary_rows = [
        ("Report", title),
        ("Data", scope_label),
        ("Generated", f"{generated_at:%Y-%m-%d %H:%M}"),
        ("Rows analyzed", f"{rows:,}"),
        ("Columns", str(cols)),
        ("Filters", "; ".join(filters_applied) or "none"),
        ("Summary written by", insights.model),
        ("Summary", insights.summary),
    ]
    if rows > max_rows:
        summary_rows.append(("Note", f"The 'Cleaned data' sheet shows only the first {max_rows:,} rows."))

    sheets: dict[str, pd.DataFrame] = {"Summary": pd.DataFrame(summary_rows, columns=["Item", "Value"])}

    sheets["KPIs"] = pd.DataFrame(
        [{"KPI": k.name, "Value": k.display, "What it means": k.description} for k in bundle.kpis]
    )
    sheets["Findings"] = pd.DataFrame(
        [
            {
                "ID": f"F{i}" if i <= 12 else "",
                "Severity": f.severity,
                "Category": f.category,
                "Title": f.title,
                "Detail": f.detail,
            }
            for i, f in enumerate(bundle.findings, start=1)
        ]
    )
    sheets["Insights"] = pd.DataFrame(
        [
            {"Type": "Insight", "Title / action": i.title, "Detail": i.explanation, "Priority": "", "Based on": ", ".join(i.based_on)}
            for i in insights.insights
        ]
        + [
            {"Type": "Recommendation", "Title / action": r.action, "Detail": r.reason, "Priority": r.priority, "Based on": ", ".join(r.based_on)}
            for r in insights.recommendations
        ]
    )
    sheets["Trends"] = pd.DataFrame(
        [
            {
                "Metric": t.metric,
                "Assessment": t.assessment,
                "Change along trend (%)": t.change_pct,
                "Latest period vs previous (%)": t.last_period_change_pct,
                "Complete periods": t.n_periods,
                "Per": t.period,
                "Aggregation": t.aggregation,
                "p-value": t.p_value,
            }
            for t in bundle.trends
        ]
    )
    an = bundle.anomalies
    sheets["Column anomalies"] = pd.DataFrame(
        [
            {
                "Column": a.column,
                "Severity": a.severity,
                "Unusual values": a.count,
                "% of rows": a.pct,
                "Lower limit": a.lower_bound,
                "Upper limit": a.upper_bound,
                "Extreme": a.extreme_count,
                "Methods": ", ".join(a.methods),
            }
            for a in an.columns
        ]
    )
    sheets["Period anomalies"] = pd.DataFrame(
        [
            {
                "Metric": a.metric,
                "Period": a.period,
                "Value": round(a.value, 2),
                "Expected": round(a.expected, 2),
                "Deviation %": a.deviation_pct,
                "Direction": a.direction,
                "Severity": a.severity,
            }
            for a in an.periods
        ]
    )
    if an.rows is not None:
        sheets["Unusual rows"] = an.rows.rows.head(200)
    sheets["Cleaning log"] = log_to_dataframe(full_cleaning.log)
    sheets["Column types"] = profiles_to_dataframe(cleaning.profiles)
    sheets["Cleaned data"] = cleaning.df.head(max_rows)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            if frame.empty and name != "Summary":
                continue
            sanitize_frame(frame).to_excel(writer, sheet_name=name, index=False)
        _format_workbook(writer)
    return buffer.getvalue()