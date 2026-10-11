"""Tests for the SQLite analysis history."""
import sqlite3

import numpy as np
import pandas as pd

from src.ai.insights import generate_insights
from src.analysis.pipeline import run_analysis
from src.cleaning.cleaner import clean_dataframe
from src.database.history import (
    FindingRow,
    HistoryStore,
    KPIRow,
    RunRecord,
    TrendRow,
    build_run_record,
)


def make_record(revenue=100.0, titles=("8 unusual values in revenue",), rows=500, scope="sales.csv"):
    return RunRecord(
        filename="sales.csv", sheet=None, scope=scope, filters="",
        rows=rows, columns=5, rows_original=rows, summary="a summary", summary_source="rules",
        kpis=[
            KPIRow("Rows analyzed", "count", None, f"{rows:,}", float(rows)),
            KPIRow("Total revenue", "total", "revenue", f"{revenue:,.0f}", revenue),
            KPIRow("Period covered", "period", "order_date", "300 days", None),
        ],
        findings=[FindingRow("high", "anomaly", t, "detail", "revenue") for t in titles],
        trends=[TrendRow("revenue", "improving", 20.0, 5.0, 11, "month", "sum", 0.01)],
    )


def test_save_and_load_a_run(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    run_id, created = store.save_run(make_record())
    assert created
    run = store.get_run(run_id)
    assert run["filename"] == "sales.csv"
    assert len(run["kpis"]) == 3
    assert run["findings"][0]["title"] == "8 unusual values in revenue"
    assert run["trends"][0]["metric"] == "revenue"
    assert run["n_findings"] == 1 and run["n_high"] == 1


def test_identical_run_is_not_saved_twice(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    first_id, first_created = store.save_run(make_record())
    second_id, second_created = store.save_run(make_record())
    assert first_created and not second_created
    assert first_id == second_id
    assert len(store.list_runs()) == 1


def test_runs_are_newest_first_and_delete_cascades(tmp_path):
    db = tmp_path / "h.db"
    store = HistoryStore(db)
    id1, _ = store.save_run(make_record(revenue=100.0))
    id2, _ = store.save_run(make_record(revenue=200.0))
    assert [r["id"] for r in store.list_runs()] == [id2, id1]

    assert store.delete_run(id1) is True
    assert store.get_run(id1) is None
    assert store.delete_run(id1) is False
    with sqlite3.connect(db) as conn:
        leftovers = conn.execute("SELECT COUNT(*) FROM run_kpis WHERE run_id = ?", (id1,)).fetchone()[0]
    assert leftovers == 0  # the KPI rows were deleted together with the run


def test_compare_two_runs(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    a, _ = store.save_run(make_record(revenue=100.0, titles=("8 unusual values in revenue", "5 duplicate rows removed")))
    b, _ = store.save_run(make_record(revenue=150.0, titles=("9 unusual values in revenue", "Column 'x' was dropped")))

    comparison = store.compare(a, b)
    revenue = comparison.kpis.set_index("KPI").loc["Total revenue"]
    assert revenue["Change"] == 50.0
    assert revenue["Change %"] == 50.0
    assert comparison.only_in_a == ["5 duplicate rows removed"]
    assert comparison.only_in_b == ["Column 'x' was dropped"]
    assert comparison.in_both == ["9 unusual values in revenue"]  # same finding, only its count changed


def test_kpi_history_across_runs(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    for revenue in (100.0, 150.0, 200.0):
        store.save_run(make_record(revenue=revenue))
    history = store.kpi_history("sales.csv", "Total revenue")
    assert history["value"].tolist() == [100.0, 150.0, 200.0]
    assert store.filenames() == ["sales.csv"]
    names = store.kpi_names("sales.csv")
    assert "Total revenue" in names and "Period covered" not in names


def test_record_built_from_a_real_analysis(tmp_path):
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
    cleaning = clean_dataframe(df)
    bundle = run_analysis(cleaning)
    insights = generate_insights(bundle, cleaning, "x.csv", use_llm=False)

    record = build_run_record("x.csv", None, "x.csv", [], cleaning, bundle, insights)
    assert len(record.kpis) == len(bundle.kpis)
    assert len(record.findings) == len(bundle.findings)
    assert record.rows == cleaning.after_shape[0]

    store = HistoryStore(tmp_path / "h.db")
    run_id, created = store.save_run(record)
    assert created
    assert len(store.get_run(run_id)["kpis"]) == len(bundle.kpis)