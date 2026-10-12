"""Time every stage of the pipeline on a large synthetic file.

Run from the project root:   python scripts\\benchmark.py            (100,000 rows)
                             python scripts\\benchmark.py 200000
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.runner import analyze_file  # noqa: E402
from src.config import settings  # noqa: E402
from src.reporting.excel_report import build_excel_report  # noqa: E402
from src.reporting.html_report import build_html_report  # noqa: E402


def make_data(rows: int) -> bytes:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "order_id": np.arange(rows),
            "order_date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, rows), unit="D"),
            "region": rng.choice(["North", "South", "East", "West", "Central"], rows),
            "product": rng.choice([f"Product {i}" for i in range(40)], rows),
            "segment": rng.choice(["Retail", "Wholesale", "Online"], rows),
            "notes": rng.choice(["ok", "late delivery", "refund requested", "gift wrapped"], rows),
        }
    )
    for i in range(10):
        df[f"metric_{i}"] = rng.gamma(2 + i, 50, rows).round(2)
    df["revenue"] = (df["metric_0"] * rng.integers(1, 9, rows)).round(2)
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    data = make_data(rows)
    size_mb = len(data) / 1e6
    print(f"Rows: {rows:,}   File size: {size_mb:.1f} MB   Upload limit: {settings.max_upload_mb} MB")
    if size_mb > settings.max_upload_mb:
        print("This file is above the upload limit. Use fewer rows.")
        return

    start = time.perf_counter()
    result = analyze_file(data, "benchmark.csv")
    analysis_seconds = time.perf_counter() - start
    print("\nAnalysis stages:")
    for stage, seconds in result.timings.items():
        print(f"  {stage:10} {seconds:8.2f} s")
    print(f"  {'total':10} {analysis_seconds:8.2f} s")

    common = dict(
        title="Benchmark", scope_label="benchmark.csv", cleaning=result.cleaning,
        full_cleaning=result.cleaning, bundle=result.bundle, insights=result.insights, filters_applied=[],
    )
    start = time.perf_counter()
    html = build_html_report(**common, raw_profiles=result.raw_profiles, chart_mode="inline")
    print(f"\nHTML report with charts: {time.perf_counter() - start:6.2f} s  ({len(html) / 1e6:.1f} MB)")
    start = time.perf_counter()
    xlsx = build_excel_report(**common)
    print(f"Excel workbook:          {time.perf_counter() - start:6.2f} s  ({len(xlsx) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()