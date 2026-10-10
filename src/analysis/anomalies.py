"""Anomaly detection: IQR, Z-score, Isolation Forest, and unusual periods."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis.trends import TrendResult
from src.profiling.type_detector import NUMERIC, ColumnProfile, columns_of_type

IQR_MULTIPLIER = 1.5
EXTREME_IQR_MULTIPLIER = 3.0
Z_THRESHOLD = 3.0
MAX_SKEW_FOR_Z = 1.0         # Z-score assumes a roughly symmetric distribution
MAX_ANOMALY_PCT = 5.0        # flagging more than this suggests natural skew, not anomalies
MIN_ROWS_ISOLATION = 50
ISOLATION_CONTAMINATION = 0.01
PERIOD_Z_THRESHOLD = 2.5
MIN_PERIODS_FOR_ANOMALY = 6


@dataclass
class ColumnAnomalies:
    column: str
    count: int
    pct: float
    lower_bound: float
    upper_bound: float
    methods: list[str]
    extreme_count: int
    severity: str                       # high | medium | low
    flagged_index: list = field(default_factory=list)
    examples: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class RowAnomalies:
    count: int
    pct: float
    columns_used: list[str]
    severity: str                       # high | medium
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    method: str = "Isolation Forest"


@dataclass
class PeriodAnomaly:
    metric: str
    period: str
    date: pd.Timestamp
    period_label: str
    value: float
    expected: float
    deviation_pct: float
    z_score: float
    direction: str                      # above | below
    severity: str                       # high | medium


@dataclass
class AnomalyResult:
    columns: list[ColumnAnomalies] = field(default_factory=list)
    rows: RowAnomalies | None = None
    periods: list[PeriodAnomaly] = field(default_factory=list)


_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def detect_column_anomalies(df: pd.DataFrame, numeric_cols: list[str]) -> list[ColumnAnomalies]:
    results: list[ColumnAnomalies] = []
    for col in numeric_cols:
        s = df[col].dropna().astype("float64")
        if len(s) < 8:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:  # more than half the values are identical: the rule cannot judge
            continue
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr

        iqr_flag = (s < lower) | (s > upper)
        flagged = iqr_flag.copy()
        methods = ["IQR"] if iqr_flag.any() else []

        if abs(float(s.skew())) <= MAX_SKEW_FOR_Z and s.std() > 0:
            z_flag = ((s - s.mean()) / s.std()).abs() > Z_THRESHOLD
            if z_flag.any():
                methods.append("Z-score")
                flagged = flagged | z_flag

        count = int(flagged.sum())
        if count == 0:
            continue
        pct = count / len(s) * 100
        extreme = int(
            ((s < q1 - EXTREME_IQR_MULTIPLIER * iqr) | (s > q3 + EXTREME_IQR_MULTIPLIER * iqr)).sum()
        )

        if pct > MAX_ANOMALY_PCT:
            severity = "low"
        elif extreme >= 1:
            severity = "high"
        else:
            severity = "medium"

        flagged_values = s[flagged]
        distance = pd.Series(
            np.where(flagged_values > q3, (flagged_values - q3) / iqr, (q1 - flagged_values) / iqr),
            index=flagged_values.index,
        ).clip(lower=0)
        top = distance.sort_values(ascending=False).head(5)
        examples = pd.DataFrame(
            {
                "row": top.index,
                "value": flagged_values.loc[top.index].values,
                "direction": np.where(flagged_values.loc[top.index] > s.median(), "high", "low"),
                "times_iqr_beyond_box": top.round(1).values,
            }
        )

        results.append(
            ColumnAnomalies(
                column=col, count=count, pct=round(pct, 2),
                lower_bound=round(lower, 4), upper_bound=round(upper, 4),
                methods=methods, extreme_count=extreme, severity=severity,
                flagged_index=list(flagged_values.index), examples=examples,
            )
        )
    results.sort(key=lambda r: (_SEVERITY_RANK[r.severity], -r.count))
    return results


def _mahalanobis_scores(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """NumPy-only fallback: flag the rows farthest from the centre of the data (accounts for correlation)."""
    values = X.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    inverse_cov = np.linalg.pinv(np.atleast_2d(np.cov(centered, rowvar=False)))
    scores = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse_cov, centered))
    k = max(1, int(round(len(values) * ISOLATION_CONTAMINATION)))
    threshold = np.sort(scores)[-k]
    return scores >= threshold, scores, "Mahalanobis distance (fallback)"


def _unusual_row_scores(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Isolation Forest if scikit-learn can be loaded, otherwise the NumPy fallback."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:  # e.g. a compiled file blocked by a Windows application-control policy
        return _mahalanobis_scores(X)
    model = IsolationForest(
        n_estimators=200, contamination=ISOLATION_CONTAMINATION, random_state=42
    )
    predictions = model.fit_predict(X)
    return predictions == -1, -model.decision_function(X), "Isolation Forest"


def detect_row_anomalies(
    df: pd.DataFrame,
    numeric_cols: list[str],
    column_anomalies: list[ColumnAnomalies] | None = None,
) -> RowAnomalies | None:
    """Find rows that are unusual across several numeric columns together."""
    cols = [c for c in numeric_cols if df[c].nunique() > 1]
    if len(cols) < 2 or len(df) < MIN_ROWS_ISOLATION:
        return None

    X = df[cols].astype("float64")
    X = X.fillna(X.median())
    mask, scores, method = _unusual_row_scores(X)
    if not mask.any():
        return None

    out = df.loc[mask].copy()
    out["anomaly_score"] = scores[mask].round(3)
    out = out.sort_values("anomaly_score", ascending=False)

    flags_by_row: dict = {}
    for ca in column_anomalies or []:
        for idx in ca.flagged_index:
            flags_by_row.setdefault(idx, []).append(ca.column)
    out["also_unusual_in"] = [", ".join(flags_by_row.get(i, [])) for i in out.index]

    severity = "high" if (out["also_unusual_in"] != "").any() else "medium"
    return RowAnomalies(
        count=int(mask.sum()),
        pct=round(float(mask.mean() * 100), 2),
        columns_used=cols,
        severity=severity,
        rows=out,
        method=method,
    )


def _format_period(ts: pd.Timestamp, label: str) -> str:
    if label == "month":
        return f"{ts:%Y-%m}"
    if label == "week":
        return f"week ending {ts:%Y-%m-%d}"
    return f"{ts:%Y-%m-%d}"


def detect_period_anomalies(trend: TrendResult) -> list[PeriodAnomaly]:
    """Periods that sit unusually far from their own trend line."""
    s = trend.series
    if s.empty or "fitted" not in s.columns or len(s) < MIN_PERIODS_FOR_ANOMALY:
        return []
    residuals = s["value"] - s["fitted"]
    sd = residuals.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return []
    z = residuals / sd

    found: list[PeriodAnomaly] = []
    for i in s.index[(z.abs() >= PERIOD_Z_THRESHOLD).to_numpy()]:
        value, expected = float(s.loc[i, "value"]), float(s.loc[i, "fitted"])
        deviation = (value - expected) / abs(expected) * 100 if expected != 0 else 0.0
        found.append(
            PeriodAnomaly(
                metric=trend.metric,
                period=_format_period(s.loc[i, "date"], trend.period),
                date=s.loc[i, "date"],
                period_label=trend.period,
                value=value,
                expected=expected,
                deviation_pct=round(deviation, 1),
                z_score=round(float(z.loc[i]), 2),
                direction="above" if value > expected else "below",
                severity="high" if abs(z.loc[i]) >= 3 else "medium",
            )
        )
    found.sort(key=lambda a: abs(a.z_score), reverse=True)
    return found


def run_anomaly_detection(
    df: pd.DataFrame, profiles: list[ColumnProfile], trends: list[TrendResult]
) -> AnomalyResult:
    numeric_cols = columns_of_type(profiles, NUMERIC)
    column_anomalies = detect_column_anomalies(df, numeric_cols)
    rows = detect_row_anomalies(df, numeric_cols, column_anomalies)
    periods = [a for t in trends for a in detect_period_anomalies(t)]
    return AnomalyResult(columns=column_anomalies, rows=rows, periods=periods)