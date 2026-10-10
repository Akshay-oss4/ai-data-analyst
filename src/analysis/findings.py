"""Turn analysis results into a ranked list of plain, structured findings.

This list is the single source of truth for the AI explanations (Phase 7),
the report (Phase 9) and the e-mail alerts (Phase 10).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from src.analysis.anomalies import AnomalyResult
from src.analysis.eda import EDAResult
from src.analysis.kpis import format_number
from src.analysis.trends import TrendResult
from src.cleaning.cleaner import CleaningResult

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Finding:
    category: str            # trend | anomaly | correlation | data_quality
    severity: str            # high | medium | low | info
    title: str
    detail: str
    column: str | None = None
    evidence: dict = field(default_factory=dict)


def build_findings(
    cleaning: CleaningResult,
    eda: EDAResult,
    trends: list[TrendResult],
    anomalies: AnomalyResult,
) -> list[Finding]:
    findings: list[Finding] = []
    n_rows = len(cleaning.df)

    # ---- Trends ----
    for t in trends:
        if t.assessment not in ("improving", "declining"):
            continue
        verb = "rose" if t.trend == "up" else "fell"
        if t.assessment == "improving":
            severity = "info"
        else:
            severity = "high" if abs(t.change_pct) >= 20 else "medium"
        detail = (
            f"{t.metric} ({t.aggregation} per {t.period}) {verb} about {abs(t.change_pct):.0f}% "
            f"along the trend line over {t.n_periods} complete {t.period}s."
        )
        if t.last_period_change_pct is not None:
            detail += (
                f" The latest complete {t.period} was {t.last_period_change_pct:+.1f}% "
                f"compared with the one before."
            )
        findings.append(
            Finding(
                "trend", severity, f"{t.metric} is {t.assessment}", detail, t.metric,
                {
                    "trend": t.trend, "change_pct": t.change_pct,
                    "last_period_change_pct": t.last_period_change_pct,
                    "periods": t.n_periods, "period": t.period, "p_value": t.p_value,
                },
            )
        )

    # ---- Unusual periods ----
    for a in anomalies.periods:
        findings.append(
            Finding(
                "anomaly", a.severity, f"Unusual {a.metric} in {a.period}",
                f"{a.metric} in {a.period} was {format_number(a.value)}, about "
                f"{abs(a.deviation_pct):.0f}% {a.direction} the {format_number(a.expected)} "
                f"expected from the trend (z-score {a.z_score:+.1f}).",
                a.metric,
                {
                    "period": a.period, "value": round(a.value, 2),
                    "expected": round(a.expected, 2), "deviation_pct": a.deviation_pct,
                    "z_score": a.z_score, "direction": a.direction,
                },
            )
        )

    # ---- Unusual values per column ----
    for ca in anomalies.columns:
        detail = (
            f"{ca.count} of {n_rows:,} values ({ca.pct:.1f}%) fall outside the normal range of "
            f"{format_number(ca.lower_bound)} to {format_number(ca.upper_bound)} "
            f"(methods: {', '.join(ca.methods)}). {ca.extreme_count} are extreme."
        )
        if ca.severity == "low":
            detail += " So many unusual values suggests the column is naturally skewed rather than containing errors."
        findings.append(
            Finding(
                "anomaly", ca.severity, f"{ca.count} unusual values in {ca.column}", detail, ca.column,
                {
                    "count": ca.count, "pct": ca.pct, "extreme_count": ca.extreme_count,
                    "lower_bound": ca.lower_bound, "upper_bound": ca.upper_bound,
                    "examples": [
                        {"value": round(float(r.value), 2), "direction": r.direction}
                        for r in ca.examples.head(3).itertuples()
                    ],
                },
            )
        )

    # ---- Unusual rows (multivariate) ----
    if anomalies.rows is not None:
        r = anomalies.rows
        detail = (
            f"{r.count} rows ({r.pct:.1f}%) look unusual when {', '.join(r.columns_used)} "
            f"are considered together ({r.method})."
        )
        if r.severity == "high":
            detail += " Some of them are also outliers in individual columns."
        findings.append(
            Finding(
                "anomaly", r.severity, f"{r.count} unusual rows across several columns", detail, None,
                {"count": r.count, "pct": r.pct, "columns_used": r.columns_used},
            )
        )

    # ---- Correlations ----
    for pair in eda.strong_correlations[:3]:
        findings.append(
            Finding(
                "correlation", "info",
                f"{pair.column_a} and {pair.column_b} move together",
                f"{pair.strength.capitalize()} {pair.direction} correlation (r = {pair.correlation}). "
                f"This shows association, not cause.",
                None,
                {"r": pair.correlation, "columns": [pair.column_a, pair.column_b]},
            )
        )

    # ---- Data quality ----
    before_rows = cleaning.before_shape[0]
    for action in cleaning.log:
        if action.step == "remove_duplicates":
            findings.append(
                Finding("data_quality", "low", f"{action.count} duplicate rows removed", action.detail, None,
                        {"count": action.count})
            )
        elif action.step == "drop_column":
            findings.append(
                Finding("data_quality", "medium", f"Column '{action.column}' was dropped", action.detail,
                        action.column, {"missing_count": action.count})
            )
        elif action.step == "fill_missing" and before_rows and action.count / before_rows >= 0.2:
            findings.append(
                Finding("data_quality", "medium", f"Many missing values filled in '{action.column}'",
                        f"{action.count} values ({action.count / before_rows * 100:.0f}% of rows). {action.detail}.",
                        action.column, {"count": action.count})
            )

    findings.sort(key=lambda f: SEVERITY_ORDER[f.severity])  # stable within a severity
    return findings


def findings_to_dicts(findings: list[Finding]) -> list[dict]:
    return [asdict(f) for f in findings]


def findings_to_dataframe(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"severity": f.severity, "category": f.category, "title": f.title, "detail": f.detail} for f in findings]
    )