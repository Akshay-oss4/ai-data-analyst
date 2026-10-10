"""Tests for dashboard filters and KPI arrows (no Streamlit needed)."""
import pyarrow as pa
import pandas as pd

from src.analysis.kpis import KPI
from src.analysis.trends import TrendResult
from src.profiling.type_detector import profile_columns
from src.ui.filters import FilterSpec, apply_filters, refresh_profiles
from src.ui.presenters import arrow_safe, kpi_delta


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2024-01-05", "2024-02-10", "2024-03-15", "2024-04-20", None, "2024-05-25"]
            ),
            "region": ["North", "South", "North", "East", "South", "North"],
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def make_trend(metric: str, last_change: float, lower_is_better: bool = False) -> TrendResult:
    return TrendResult(
        metric=metric, date_column="order_date", period="month", aggregation="sum",
        n_periods=12, trend="up", assessment="improving", change_pct=20.0,
        last_period_change_pct=last_change, p_value=0.01, r_squared=0.9,
        lower_is_better=lower_is_better,
    )


def test_date_filter_narrows_rows():
    spec = FilterSpec(
        date_column="order_date", start=pd.Timestamp("2024-02-01"), end=pd.Timestamp("2024-04-30")
    )
    out, applied = apply_filters(make_df(), spec)
    assert len(out) == 3
    assert len(applied) == 1 and "order_date" in applied[0]


def test_full_date_range_keeps_rows_with_missing_dates():
    spec = FilterSpec(
        date_column="order_date", start=pd.Timestamp("2024-01-05"), end=pd.Timestamp("2024-05-25")
    )
    out, applied = apply_filters(make_df(), spec)
    assert len(out) == 6
    assert applied == []


def test_category_filter_and_description():
    out, applied = apply_filters(make_df(), FilterSpec(categories={"region": ["North"]}))
    assert len(out) == 3
    assert "region" in applied[0] and "North" in applied[0]


def test_filter_with_no_matches_gives_empty_result():
    out, _ = apply_filters(make_df(), FilterSpec(categories={"region": ["Nowhere"]}))
    assert out.empty


def test_refresh_profiles_updates_counts():
    df = make_df()
    profiles = profile_columns(df)
    subset = df[df["region"] == "North"].reset_index(drop=True)
    refreshed = refresh_profiles(profiles, subset)
    region = next(p for p in refreshed if p.name == "region")
    assert region.unique_count == 1
    assert len(refreshed) == len(profiles)


def test_kpi_delta_arrows():
    up = make_trend("revenue", 5.3)
    delta, color, _ = kpi_delta(KPI("Total revenue", 100.0, "100", "total", "revenue"), [up])
    assert delta == "+5.3% vs previous month"
    assert color == "normal"

    # An average of an additive metric has no arrow (the trend is about the total)
    assert kpi_delta(KPI("Average revenue", 5.0, "5", "average", "revenue"), [up])[0] is None

    # For cost, a fall is good news, so the colour is inverted
    cost = make_trend("cost", -3.04, lower_is_better=True)
    delta, color, _ = kpi_delta(KPI("Total cost", 100.0, "100", "total", "cost"), [cost])
    assert delta == "-3.0% vs previous month"
    assert color == "inverse"
    
def test_arrow_safe_converts_mixed_columns():
    df = pd.DataFrame({"mixed": [1, "a", 2.5, None], "nums": [1, 2.5, 3, 4]})
    safe = arrow_safe(df)
    pa.Table.from_pandas(safe)  # must not raise
    assert safe["mixed"].tolist()[:3] == ["1", "a", "2.5"]
    assert safe["nums"].tolist() == [1.0, 2.5, 3.0, 4.0]