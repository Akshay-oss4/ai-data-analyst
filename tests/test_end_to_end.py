"""The whole pipeline on every sample file."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from e2e_helpers import build_outputs

from src.analysis.runner import AnalysisError, analyze_file

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
SAMPLE_FILES = [
    p for p in sorted(SAMPLE_DIR.glob("*"))
    if p.suffix in {".csv", ".xlsx"} and p.name != "header_only.csv"
]


@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
def test_sample_file_goes_through_the_whole_pipeline(path, tmp_path):
    result = analyze_file(path.read_bytes(), path.name)
    assert not result.cleaning.df.empty
    assert result.bundle.kpis
    assert result.insights.summary
    assert set(result.timings) >= {"load", "validate", "profile", "clean", "analyze", "insights"}

    html, xlsx, run_id = build_outputs(result, path.name, tmp_path)
    assert "Executive summary" in html
    assert xlsx[:2] == b"PK"  # an .xlsx file is a zip archive
    assert run_id >= 1


def test_header_only_sample_is_rejected_politely():
    path = SAMPLE_DIR / "header_only.csv"
    if not path.exists():
        pytest.skip("run scripts/generate_sample_data.py first")
    with pytest.raises(AnalysisError):
        analyze_file(path.read_bytes(), path.name)


def test_analysis_is_deterministic():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=300, freq="D"),
            "revenue": rng.normal(100, 10, 300),
            "units": rng.integers(1, 20, 300),
            "region": rng.choice(["North", "South"], 300),
        }
    )
    data = df.to_csv(index=False).encode("utf-8")
    first, second = analyze_file(data, "x.csv"), analyze_file(data, "x.csv")
    assert [k.display for k in first.bundle.kpis] == [k.display for k in second.bundle.kpis]
    assert [f.title for f in first.bundle.findings] == [f.title for f in second.bundle.findings]