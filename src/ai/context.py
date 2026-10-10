"""Build the compact FACT SHEET that the language model is allowed to use."""
from __future__ import annotations

from src.analysis.findings import Finding
from src.analysis.pipeline import AnalysisBundle
from src.cleaning.cleaner import CleaningResult


def build_fact_sheet(
    bundle: AnalysisBundle,
    cleaning: CleaningResult,
    filename: str,
    max_findings: int = 12,
) -> tuple[str, list[tuple[str, Finding]]]:
    """Return (fact_sheet_text, [(finding_id, finding), ...])."""
    numbered = [(f"F{i}", f) for i, f in enumerate(bundle.findings[:max_findings], start=1)]
    before_rows, before_cols = cleaning.before_shape
    after_rows, after_cols = cleaning.after_shape

    lines = [
        "DATASET",
        f"- File: {filename}",
        f"- Rows analyzed: {after_rows:,} (original file: {before_rows:,}); "
        f"columns: {after_cols} (original: {before_cols})",
    ]

    if cleaning.log:
        lines.append("CLEANING DONE")
        for action in cleaning.log[:8]:
            where = f" ({action.column})" if action.column else ""
            lines.append(f"- {action.step.replace('_', ' ')}{where}: {action.count}")

    lines.append("KPIS")
    for kpi in bundle.kpis[:12]:
        lines.append(f"- {kpi.name}: {kpi.display}")

    if bundle.trends:
        lines.append("TRENDS (per metric)")
        for t in bundle.trends:
            line = f"- {t.metric}: {t.assessment}"
            if t.change_pct is not None:
                line += f", {t.change_pct:+.0f}% along the trend line over {t.n_periods} complete {t.period}s"
            lines.append(line)

    lines.append("FINDINGS (most important first)")
    if not numbered:
        lines.append("- none")
    for fid, f in numbered:
        lines.append(f"[{fid}] ({f.severity}, {f.category}) {f.title}. {f.detail}")

    return "\n".join(lines), numbered