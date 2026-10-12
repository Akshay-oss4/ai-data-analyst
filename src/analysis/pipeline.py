"""One call that runs the whole analysis engine on a cleaned dataset."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.analysis.anomalies import AnomalyResult, run_anomaly_detection
from src.analysis.eda import EDAResult, run_eda
from src.analysis.findings import Finding, build_findings
from src.analysis.kpis import KPI, compute_kpis
from src.analysis.trends import TrendResult, analyze_trends
from src.cleaning.cleaner import CleaningResult


@dataclass
class AnalysisBundle:
    eda: EDAResult
    kpis: list[KPI]
    trends: list[TrendResult]
    anomalies: AnomalyResult
    findings: list[Finding]


def run_analysis(cleaning: CleaningResult) -> AnalysisBundle:
    df, profiles = cleaning.df, cleaning.profiles
    with np.errstate(all="ignore"):  # values near 1e300 can overflow harmlessly in statistics
        eda = run_eda(df, profiles)
        kpis = compute_kpis(df, profiles)
        trends = analyze_trends(df, profiles)
        anomalies = run_anomaly_detection(df, profiles, trends)
        findings = build_findings(cleaning, eda, trends, anomalies)
    return AnalysisBundle(eda=eda, kpis=kpis, trends=trends, anomalies=anomalies, findings=findings)