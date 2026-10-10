"""Tests for trends, anomaly detection, findings and the pipeline."""
import numpy as np
import pandas as pd

from src.analysis.anomalies import (
    detect_column_anomalies,
    detect_period_anomalies,
    detect_row_anomalies,
)
from src.analysis.pipeline import run_analysis
from src.analysis.trends import analyze_trend, lower_is_better
from src.cleaning.cleaner import clean_dataframe


def daily_df(metric: str, values, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"order_date": dates, metric: values})


def test_lower_is_better():
    assert lower_is_better("cost")
    assert lower_is_better("churn_rate")
    assert not lower_is_better("revenue")


def test_trend_improving_when_growing():
    df = daily_df("revenue", 100 + 0.25 * np.arange(400))
    t = analyze_trend(df, "order_date", "revenue")
    assert t.trend == "up"
    assert t.assessment == "improving"
    assert t.change_pct > 5
    assert t.period == "month"
    assert t.n_periods == 13


def test_falling_cost_is_improving():
    df = daily_df("cost", 200 - 0.25 * np.arange(400))
    t = analyze_trend(df, "order_date", "cost")
    assert t.trend == "down"
    assert t.lower_is_better
    assert t.assessment == "improving"


def test_flat_metric_is_stable():
    rng = np.random.default_rng(0)
    df = daily_df("customer_rating", rng.normal(100, 1, 400))
    assert analyze_trend(df, "order_date", "customer_rating").assessment == "stable"


def test_insufficient_data():
    df = daily_df("revenue", [1.0, 2.0, 3.0])
    assert analyze_trend(df, "order_date", "revenue").assessment == "insufficient data"


def test_incomplete_periods_dropped():
    dates = pd.date_range("2024-01-15", "2024-12-10")
    df = pd.DataFrame({"order_date": dates, "revenue": 100.0 + np.arange(len(dates))})
    # January (starts on the 15th) and December (ends on the 10th) are partial
    assert analyze_trend(df, "order_date", "revenue").n_periods == 10


def test_column_anomalies_detected():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(100, 5, 200)})
    df.loc[[10, 50, 90], "x"] = 1000.0
    result = detect_column_anomalies(df, ["x"])
    assert len(result) == 1
    ca = result[0]
    assert {10, 50, 90} <= set(ca.flagged_index)
    assert ca.extreme_count >= 3
    assert ca.severity == "high"


def test_no_anomalies_in_uniform_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.uniform(0, 1, 200)})
    assert detect_column_anomalies(df, ["x"]) == []


def test_isolation_forest_finds_joint_outlier():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 300)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.1, 300)})
    df.loc[7, ["a", "b"]] = [3.0, -3.0]  # fine on each axis alone, impossible together
    result = detect_row_anomalies(df, ["a", "b"])
    assert result is not None
    assert 7 in result.rows.index


def test_period_anomaly_detected():
    df = daily_df("revenue", np.full(400, 100.0))
    june = (df["order_date"] >= "2024-06-01") & (df["order_date"] < "2024-07-01")
    df.loc[june, "revenue"] = 300.0
    trend = analyze_trend(df, "order_date", "revenue")
    anomalies = detect_period_anomalies(trend)
    assert any(a.period == "2024-06" and a.direction == "above" for a in anomalies)


def test_full_pipeline_and_findings():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "revenue": 100 + 0.25 * np.arange(n) + rng.normal(0, 3, n),
            "units": rng.integers(1, 20, n),
            "region": rng.choice(["North", "South", "East"], n),
        }
    )
    df.loc[[20, 120, 220], "revenue"] = 1000.0
    bundle = run_analysis(clean_dataframe(df))

    assert any(t.metric == "revenue" and t.assessment == "improving" for t in bundle.trends)
    assert any(f.category == "anomaly" for f in bundle.findings)
    rank = {"high": 0, "medium": 1, "low": 2, "info": 3}
    order = [rank[f.severity] for f in bundle.findings]
    assert order == sorted(order)