"""Tests for selectors, EDA, KPIs and charts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis.charts import category_bar_chart, histogram_chart, time_series_chart
from src.analysis.eda import aggregate_over_time, describe_gap, run_eda
from src.analysis.kpis import compute_kpis, format_number
from src.analysis.metric_selectors import is_additive, pick_primary_date_column, pick_primary_metrics
from src.profiling.type_detector import profile_columns


def make_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(100, 15, n)
    return pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "revenue": x,
            "units": rng.integers(1, 20, n),
            "customer_rating": rng.normal(4, 0.5, n).round(1),
            "region": rng.choice(["North", "South", "East"], n),
            "revenue_copy": x * 2 + rng.normal(0, 1, n),
        }
    )


def test_is_additive():
    assert is_additive("revenue")
    assert is_additive("units")
    assert is_additive("Total Price")
    assert not is_additive("unit_price")
    assert not is_additive("customer_rating")
    assert not is_additive("Age")


def test_pick_primary_metrics_and_date_column():
    profiles = profile_columns(make_df())
    assert pick_primary_metrics(profiles, k=2) == ["revenue", "revenue_copy"]
    assert pick_primary_date_column(profiles) == "order_date"


def test_numeric_and_categorical_summaries():
    df = make_df()
    eda = run_eda(df, profile_columns(df))
    assert set(eda.numeric_columns) == {"revenue", "units", "customer_rating", "revenue_copy"}
    row = eda.numeric_summary.set_index("column").loc["revenue"]
    assert abs(row["median"] - df["revenue"].median()) < 0.01
    assert abs(eda.categorical_summaries["region"]["pct"].sum() - 100) < 0.5


def test_strong_correlation_detected():
    df = make_df()
    eda = run_eda(df, profile_columns(df))
    assert any(
        {p.column_a, p.column_b} == {"revenue", "revenue_copy"}
        and p.direction == "positive"
        and p.strength == "very strong"
        for p in eda.strong_correlations
    )


def test_aggregate_over_time_monthly_and_daily():
    year = pd.DataFrame({"order_date": pd.date_range("2024-01-01", periods=365), "revenue": 1.0})
    ts, label = aggregate_over_time(year, "order_date", "revenue")
    assert label == "month"
    assert len(ts) == 12
    assert ts.iloc[0]["value"] == 31

    short = pd.DataFrame({"order_date": pd.date_range("2024-01-01", periods=20), "revenue": 1.0})
    ts2, label2 = aggregate_over_time(short, "order_date", "revenue")
    assert label2 == "day"
    assert len(ts2) == 20


def test_describe_gap():
    assert describe_gap(1) == "daily"
    assert describe_gap(7) == "weekly"
    assert describe_gap(30) == "monthly"


def test_datetime_info_frequency():
    df = make_df()
    info = run_eda(df, profile_columns(df)).datetime_info[0]
    assert info["frequency"] == "daily"
    assert info["span_days"] == 119


def test_compute_kpis():
    df = make_df()
    kpis = compute_kpis(df, profile_columns(df))
    names = [k.name for k in kpis]
    assert kpis[0].name == "Rows analyzed" and kpis[0].value == 120
    assert "Total revenue" in names
    assert "Average units" in names
    assert "Top region" in names


def test_format_number():
    assert format_number(1_234_567) == "1.23M"
    assert format_number(12_500) == "12.5K"
    assert format_number(42) == "42"
    assert format_number(1234) == "1,234"
    assert format_number(3.14159) == "3.14"


def test_charts_build():
    df = make_df()
    ts, label = aggregate_over_time(df, "order_date", "revenue")
    assert isinstance(histogram_chart(df, "revenue"), go.Figure)
    assert isinstance(category_bar_chart(df, "region"), go.Figure)
    assert isinstance(time_series_chart(ts, "revenue", label), go.Figure)
