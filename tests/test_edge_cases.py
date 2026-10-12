"""Awkward files that must never crash the pipeline."""
import io

import numpy as np
import pandas as pd
import pytest
from e2e_helpers import build_outputs

from src.analysis.anomalies import detect_row_anomalies
from src.analysis.charts import histogram_chart
from src.analysis.kpis import format_number
from src.analysis.runner import AnalysisError, analyze_file
from src.cleaning.cleaner import clean_dataframe
from src.ingestion.loader import FileLoadError, load_file
from src.reporting.excel_report import sanitize_frame

rng = np.random.default_rng(5)


def csv_bytes(df: pd.DataFrame, sep: str = ",") -> bytes:
    return df.to_csv(index=False, sep=sep).encode("utf-8")


def inf_csv() -> bytes:
    lines = ["x,y"] + [f"{i},{i * 2}" for i in range(1, 19)] + ["inf,40", "-inf,42"]
    return "\n".join(lines).encode()


OK_CASES = {
    "one_row": b"a,b,c\n1,2,3\n",
    "single_numeric_column": csv_bytes(pd.DataFrame({"x": rng.normal(size=40)})),
    "all_text": csv_bytes(
        pd.DataFrame(
            {"a": [f"word{i}" for i in range(40)], "b": ["some free text here"] * 40, "c": rng.choice(["p", "q"], 40)}
        )
    ),
    "constant_columns": csv_bytes(pd.DataFrame({"a": [1] * 30, "b": ["k"] * 30})),
    "empty_column_next_to_data": csv_bytes(pd.DataFrame({"v": rng.normal(size=30), "empty": [np.nan] * 30})),
    "dates_only": csv_bytes(pd.DataFrame({"d": pd.date_range("2024-01-01", periods=40)})),
    "huge_numbers": csv_bytes(
        pd.DataFrame(
            {
                "a": [1e300 * (i + 1) for i in range(60)],
                "b": rng.normal(size=60) * 1e290,
                "c": rng.normal(size=60),
            }
        )
    ),
    "infinite_values": inf_csv(),
    "utf16_file": "a;b\n1;2\n3;4\n5;6\n".encode("utf-16"),
    "quoted_newlines": b'name,note\nA,"line1\nline2"\nB,plain\nC,"x\ny"\n',
    "duplicate_headers": b"a,a,b\n1,2,3\n4,5,6\n",
    "unicode_headers": csv_bytes(
        pd.DataFrame(
            {"Ürün": [f"çay{i % 3}" for i in range(30)], "价格": rng.normal(10, 2, 30), "💰": rng.integers(1, 9, 30)}
        )
    ),
    "wide_file": csv_bytes(pd.DataFrame(rng.normal(size=(60, 120)), columns=[f"c{i}" for i in range(120)])),
    "control_characters": ("name,note\n" + "\n".join(f"n{i},bad\x01text{i}" for i in range(30))).encode(),
    "formula_like_text": csv_bytes(
        pd.DataFrame({"cell": ["=1+1", "+cmd", "@x", "-5text", "plain"] * 6, "n": rng.normal(size=30)})
    ),
    "very_long_text": csv_bytes(
        pd.DataFrame({"t": [f"{i}" + "x" * 40000 for i in range(25)], "n": rng.normal(size=25)})
    ),
    "mixed_types_column": csv_bytes(pd.DataFrame({"m": [1, "two", 3, "four"] * 10, "n": rng.normal(size=40)})),
    "decimal_comma_numbers": csv_bytes(
        pd.DataFrame(
            {"v": [f"{x:.2f}".replace(".", ",") for x in rng.normal(10, 2, 30)], "k": ["a", "b", "c"] * 10}
        ),
        sep=";",
    ),
    "negative_numbers_only": csv_bytes(pd.DataFrame({"v": -rng.random(40) * 100, "w": -rng.random(40)})),
    "yes_no_columns": csv_bytes(
        pd.DataFrame(
            {"flag": rng.choice(["Yes", "No"], 40), "ok": rng.choice([True, False], 40), "n": rng.normal(size=40)}
        )
    ),
}

FAIL_CASES = {
    "header_only": b"a,b,c\n",
    "all_empty_tokens": b"a,b\nN/A,null\n-,\n,?\n",
    "binary_file": b"PK\x03\x04\x00\x00\x00\x00binary",
}


@pytest.mark.parametrize("name", list(OK_CASES))
def test_unusual_file_runs_end_to_end(name, tmp_path):
    result = analyze_file(OK_CASES[name], f"{name}.csv")
    html, xlsx, run_id = build_outputs(result, f"{name}.csv", tmp_path)
    assert "Executive summary" in html
    assert xlsx[:2] == b"PK"
    assert run_id >= 1


@pytest.mark.parametrize("name", list(FAIL_CASES))
def test_unusable_file_is_rejected_politely(name):
    with pytest.raises((FileLoadError, AnalysisError)):
        analyze_file(FAIL_CASES[name], f"{name}.csv")


@pytest.mark.parametrize("name", ["one_row", "huge_numbers", "wide_file"])
def test_charts_render_for_edge_cases(name, tmp_path):
    result = analyze_file(OK_CASES[name], f"{name}.csv")
    html, _, _ = build_outputs(result, f"{name}.csv", tmp_path, chart_mode="inline")
    assert "Plotly.newPlot" in html


def test_infinite_values_become_missing_and_are_filled():
    df = pd.DataFrame({"x": [1.0, np.inf, 3.0, -np.inf, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    result = clean_dataframe(df)
    assert np.isfinite(result.df["x"]).all()
    assert any(a.step == "replace_infinite" for a in result.log)


def test_utf16_file_is_decoded():
    loaded = load_file("a;b\n1;2\n".encode("utf-16"), "x.csv")
    assert loaded.encoding == "utf-16"
    assert list(loaded.df.columns) == ["a", "b"]


def test_excel_export_survives_control_characters_and_very_long_text():
    frame = pd.DataFrame({"t": ["bad\x01text", "x" * 40000, "=1+1"]})
    safe = sanitize_frame(frame)
    assert "\x01" not in safe["t"][0]
    assert len(safe["t"][1]) <= 32_001
    assert safe["t"][2].startswith("'=")
    safe.to_excel(io.BytesIO(), index=False)  # must not raise


def test_histogram_uses_a_sample_for_large_data():
    df = pd.DataFrame({"v": np.random.default_rng(0).normal(size=60_000)})
    assert "sample" in histogram_chart(df, "v").layout.title.text


def test_format_number_handles_very_large_values():
    assert format_number(5e12) == "5.00T"
    assert format_number(1e20) == "1.00e+20"


def test_row_anomalies_skip_columns_with_astronomical_values():
    gen = np.random.default_rng(1)
    df = pd.DataFrame(
        {"a": gen.normal(size=100) * 1e300, "b": gen.normal(size=100), "c": gen.normal(size=100)}
    )
    result = detect_row_anomalies(df, ["a", "b", "c"])
    assert result is None or "a" not in result.columns_used